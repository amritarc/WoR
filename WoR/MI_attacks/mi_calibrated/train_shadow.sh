#!/usr/bin/env bash

# ~1098 MB GPU memory
# N_train=60000
GPU=1

MODEL="LeNet5"
for SEED in 0 1 2
do
    for N_train in 10000 60000
    do
        python train_shadow.py \
            --dataset 'mnist' \
            --model $MODEL \
            --img_size 32 \
            --batch_size 100 \
            --n_epochs 100 \
            --N_train $N_train \
            --seed $SEED \
            --gpu $GPU & 
    done 
    wait
done
