#!/usr/bin/env bash

cd .. 

# 1863 MB GPU memory B=64, 1987 MB B=1K
N_train=30000  # n_sample for training ori
N_forge=30000  # n_sample for forging
n=1600  # number of candidates
M=200  # number of random trials
# forge_type="model"  # "paper", "grad", "model", grad is the worst, so just compare paper and model

GPU=0

for forge_type in "paper" "model"
do
    python MI_forge_half_from_anotherhalf.py \
        --model 'LeNet5' \
        --img_size 32 \
        --batch_size 64 \
        --lr 0.05 \
        --n_epochs 100 \
        --ckpt_every 2 \
        --N_train $N_train \
        --N_forge $N_forge \
        --n $n \
        --M $M \
        --forge \
        --forge_type $forge_type \
        --gpu "$GPU" &
    GPU=$((1-GPU))
done

cd scripts || exit