#!/usr/bin/env bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cd .. 

NUM_EXP=16
export CUDA_VISIBLE_DEVICES='0' 

# for ID in {0..15}; do
#     python3 -u train.py \
#         --dataset=mnist \
#         --dataset_size 60000 \
#         --lr 0.001 \
#         --batch 64 \
#         --epochs=20 \
#         --save_steps=10 \
#         --arch cnn32-3-max \
#         --num_experiments ${NUM_EXP} \
#         --seed 0 \
#         --expid "${ID}" \
#         --logdir exp/mnist-60k &> logs/log_"${ID}"
# done


for ID in {2..15}; do
    python3 -u train.py \
        --dataset=cifar10 \
        --dataset_size 50000 \
        --lr 0.1 \
        --batch 256 \
        --epochs=100 \
        --save_steps=10 \
        --arch wrn28-2 \
        --num_experiments ${NUM_EXP} \
        --seed 0 \
        --expid "${ID}" \
        --logdir exp/cifar10-50k &> logs/log_"${ID}"
done

cd scripts || exit