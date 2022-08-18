# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from cta.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment

FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class FixMatch(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            self.tmp.s1_epoch_count = 0
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, gen_labeled, gen_unlabeled, FLAGS.labeled_num,
                                    use_dash_policy = True,
                                    append_xeu_loss = self.tmp.s1_epoch_count < FLAGS.dt_start_epoch,
                                    total_train_num = FLAGS.total_train_num)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
                if self.tmp.s1_epoch_count < FLAGS.dt_start_epoch - 1:
                    while len(self.tmp.xeu_losses) > 0:
                        self.tmp.xeu_losses.pop()
                self.tmp.s1_epoch_count += 1
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def model(self, batch, lr, wd, wu, confidence, uratio, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels
        xeu_losses_in = tf.placeholder(tf.float32, [int(FLAGS.total_train_num - FLAGS.labeled_num)], 'xeu_losses_in')

        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        dynamic_t = tf.to_int32(tf.to_float(self.step) / FLAGS.train_kimg)
        def weak_dynamic_th_fn1(): return 500.0
        def weak_dynamic_th_fn2(): return tf.reduce_mean(xeu_losses_in) * \
                                          tf.pow(FLAGS.xeu_select_gamma, -1.0 * tf.to_float(tf.to_int32((dynamic_t - 1)/FLAGS.drop_step)))
        weak_dynamic_select_th = tf.cond(
            tf.to_float(self.step) / FLAGS.train_kimg < FLAGS.dt_start_epoch,
            weak_dynamic_th_fn1,
            weak_dynamic_th_fn2
        )
        tf.summary.scalar('monitors/dynamic_t', dynamic_t)
        tf.summary.scalar('monitors/weak_select_th', weak_dynamic_select_th)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio + 1)
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # Labeled cross-entropy
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)

        # append additional supervised loss
        loss_xe = tf.reduce_mean(loss_xe)
        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        temperature = FLAGS.temperature
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak / temperature))
        weak_dynamic_select_th_with_min_value = tf.math.maximum(weak_dynamic_select_th, FLAGS.min_select_th)
        tf.summary.scalar('monitors/weak_select_th_with_min_value', weak_dynamic_select_th_with_min_value)

        pseudo_labels_one_hot = tf.argmax(pseudo_labels, axis=1)
        loss_pseudo_labels = tf.stop_gradient(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_labels_one_hot,
                                                                                                logits=logits_weak))
        pseudo_mask = tf.to_float(loss_pseudo_labels < weak_dynamic_select_th_with_min_value)

        # when dynamic threshold decay to min_select_th, use onehot label for loss
        use_one_hot_loss_flag = tf.cond(weak_dynamic_select_th > FLAGS.min_select_th, lambda: 0.0, lambda: 1.0)
        one_hot_loss_xeu_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_labels_one_hot,
                                                                              logits=logits_strong)
        loss_xeu_vec = tf.nn.softmax_cross_entropy_with_logits(labels=pseudo_labels,
                                                           logits=logits_strong)
        tf.summary.scalar('monitors/weak_mask_count', tf.reduce_sum(pseudo_mask))
        tf.summary.scalar('monitors/weak_mask_ratio', tf.reduce_mean(pseudo_mask))
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))

        one_hot_loss_xeu = tf.math.divide_no_nan(
            tf.reduce_sum(one_hot_loss_xeu_vec * pseudo_mask), tf.reduce_sum(pseudo_mask)
        ) * 0.9
        loss_xeu = tf.math.divide_no_nan(tf.reduce_sum(loss_xeu_vec * pseudo_mask), tf.reduce_sum(pseudo_mask)) * 0.9

        tf.summary.scalar('losses/xeu', loss_xeu)
        tf.summary.scalar('losses/one_hot_xeu', one_hot_loss_xeu)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        total_loss_xeu = use_one_hot_loss_flag * one_hot_loss_xeu + (1. - use_one_hot_loss_flag) * loss_xeu
        tf.summary.scalar('losses/total_loss_xeu', total_loss_xeu)
        total_loss = loss_xe + wu * total_loss_xeu + wd * loss_wd
        tf.summary.scalar('losses/total_loss', total_loss)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            total_loss, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op, loss_xeu_vec=loss_xeu_vec,
            xeu_losses_in=xeu_losses_in,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FixMatch(
        os.path.join(FLAGS.train_dir, dataset.name, FixMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    print('tf.test.is_gpu_available() = {}'.format(tf.test.is_gpu_available()))
    print('tested tf GPU')
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.001, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    flags.DEFINE_string('pretrained_path', '', 'Pretrained path.')
    flags.DEFINE_integer('labeled_num', 4000, 'labeled_num')
    flags.DEFINE_integer('total_train_num', 50000, 'total_train_num')
    flags.DEFINE_float('xeu_select_gamma', 1.004, 'xeu_select_gamma')
    flags.DEFINE_integer('dt_start_epoch', 1, 'select specific samples since this epoch')
    flags.DEFINE_integer('drop_step', 1, 'drop_step')
    flags.DEFINE_float('min_select_th', 0.05, 'min_select_th')
    flags.DEFINE_float('temperature', 1.0, 'temperature')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
