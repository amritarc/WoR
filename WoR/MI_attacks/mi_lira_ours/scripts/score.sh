#!/usr/bin/env bash

cd ..

# python3 score.py exp/mnist-60k/
python3 score.py exp/cifar10-50k/
python3 plot.py

cd scripts || exit