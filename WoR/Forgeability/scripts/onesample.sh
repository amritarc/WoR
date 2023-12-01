#!/usr/bin/env bash

cd .. 

# 1863 MB GPU memory
# delete=0
N_train=3200
n=1600
M=200
GPU=1
forge_type="paper"

for delete in 1 2 4 5
do  
    python MI_forge_onesample.py \
        --model 'LeNet5' \
        --img_size 32 \
        --N_train $N_train \
        --n $n \
        --M $M \
        --forge \
        --forge_type $forge_type \
        --delete "$delete" \
        --gpu $GPU & 
    # wait
done

cd scripts || exit