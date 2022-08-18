source ./setup-env.sh
CUDA_VISIBLE_DEVICES=0,1 python dash.py --filters=32 --dataset=svhn_noextra.1@40-1 --train_dir ./experiments/dash_start010_gamma1.27_wd5e-4_t0.5_minth0.05_dstep9_exp --wd 5e-4 --xeu_select_gamma 1.27 --temperature 0.5 --labeled_num 40 --min_select_th 0.05 --drop_step 9 --lr 0.06 --dt_start_epoch 10 --total_train_num 73257
