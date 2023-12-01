#!/usr/bin/env bash

cd .. 

# ~1863 MB GPU memory
N_train=60000
M=200
GPU=0

for SEED in {1..5}
do
    python MI_forge_all.py \
        --model 'LeNet5' \
        --img_size 32 \
        --lr 0.01 \
        --batch_size 100 \
        --n_epochs 20 \
        --seed $SEED \
        --N_train $N_train \
        --M $M \
        --num_del 100 \
        --splitK 5 \
        --valid_every 1000000000 \
        --gpu $GPU &
    GPU=$((1-GPU))
done

cd scripts || exit