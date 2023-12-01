#!/usr/bin/env bash

cd ..

export CUDA_VISIBLE_DEVICES='1'
# python3 inference.py \
#     --dataset=mnist \
#     --dataset_size 60000 \
#     --logdir=exp/mnist-60k/

python3 inference.py \
    --dataset=cifar10 \
    --dataset_size 50000 \
    --logdir=exp/cifar10-50k/


cd scripts || exit