#!/usr/bin/env bash

cd .. 

# ~1863 MB GPU memory
N_train=60000
M=400
GPU=1

for splitK in 5 10
do
    python MI_forge_all.py \
        --model 'LeNet5' \
        --img_size 32 \
        --lr 0.01 \
        --batch_size 100 \
        --n_epochs 20 \
        --forge \
        --N_train $N_train \
        --M $M \
        --num_del 100 \
        --splitK $splitK \
        --gpu $GPU &
    GPU=$((1-GPU))
done

cd scripts || exit