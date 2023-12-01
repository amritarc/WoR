#!/usr/bin/env bash

cd .. 

# N=10000
# Ndel=1  # 200 epochs for del 1 and 20 for del 10/100
B=100
M=200
K=5
for Ndel in 1 10 100
do 
    if [ $Ndel == 1 ]; then
        N=10000
        EPOCHS=200
    elif [ $Ndel == 10 ]; then
        N=60000
        EPOCHS=20
    elif [ $Ndel == 100 ]; then
        N=60000
        EPOCHS=20
    fi

    ind_file="tensorboard_delete_${Ndel}/mnist-LeNet5-Ntrain${N}-K${K}-M${M}/forge.pkl"

    python batch_test.py \
        --N ${N} \
        --Ndel ${Ndel} \
        --epochs ${EPOCHS} \
        --batchsize ${B} \
        --first_n_exp 100 \
        --ind_file ${ind_file}
done
cd scripts || exit