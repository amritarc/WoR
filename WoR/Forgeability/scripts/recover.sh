#!/usr/bin/env bash

cd .. 

# ~1863 MB GPU memory
N_train=60000
M=400
splitK=10
# DELETE="1:2"

export CUDA_VISIBLE_DEVICES="0,1"
GPU=0

for num_del in 100
do
    DELTA=$((N_train/num_del/6))
    for i in {1..6}
    do  
        
        DELETE=$((i*DELTA-DELTA)):$((i*DELTA))

        python recover_forged_models.py \
            --model 'LeNet5' \
            --img_size 32 \
            --lr 0.01 \
            --batch_size 100 \
            --n_epochs 20 \
            --delete "${DELETE}" \
            --N_train $N_train \
            --M $M \
            --num_del $num_del \
            --splitK $splitK \
            --gpu $GPU
        # GPU=$((1-GPU))

        # sleep 60

        # if [ $((i%2)) == 0 ]; then
        #     GPU=0
        #     wait
        # fi
    done
done 

cd scripts || exit