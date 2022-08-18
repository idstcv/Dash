# Dash

TensorFlow implementation for our ICML'21 paper: "Dash: Semi-Supervised Learning with Dynamic Thresholding"

## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Get FixMatch
Our code based on [FixMatch](https://github.com/google-research/fixmatch). Get FixMatch codebase and replace files with the same path. 
(cta/lib/train.py, libml/train.py)

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the Dash"

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

## Run

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training a Dash with 32 filters on svhn_noextra shuffled with `seed=1`, 40 labeled samples and 1
validation sample:
```bash
sh start_train.sh
```

## Monitor training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir ./experiments
```

## Cite this work

```
@inproceedings{xu2021dash,
    title={Dash: Semi-Supervised Learning with Dynamic Thresholding},
    author={Xu, Yi and Shang, Lei and Ye, Jinxing and Qian, Qi and Li, Yu-Feng and Sun, Baigui and Li, Hao and Jin, Rong},
    booktitle={International Conference on Machine Learning},
    year={2021},
}
```
