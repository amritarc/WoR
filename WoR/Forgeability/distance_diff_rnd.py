import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

from MI_forge_onesample import (
    calc_model_dist, LeNet5
)

from modeldist.compare_model_dist import compute_params



if __name__ == '__main__':
    dataset = 'mnist'
    Ntrain = 60000
    num_del = 10
    n_epochs = 20
    model = "LeNet5"

    net_0 = LeNet5(10)
    net_seed = deepcopy(net_0)

    checkpoint0 = torch.load(os.path.join(
        'tensorboard_delete_{}'.format(num_del),
        '{}-{}-Ntrain{}-K5-M200'.format(dataset, model, Ntrain),
        'model_ori_state_dict.pkl'
    ), map_location='cpu')
    net_0.load_state_dict(checkpoint0['model_ori_state_dict'])
    n_params = compute_params(net_0)

    dists = []
    for seed in range(1, 6):
        checkpoint_seed = torch.load(os.path.join(
            'tensorboard_randomseed',
            '{}-{}-Ntrain{}-seed{}'.format(dataset, model, Ntrain, seed), 
            'model_ori_state_dict-{}.pkl'.format(n_epochs)
        ), map_location='cpu')

        net_seed.load_state_dict(checkpoint_seed['model_ori_state_dict'])

        dists.append(calc_model_dist(net_0, net_seed))
        
    dists = np.log10(dists / n_params)
    print('distances between randomly init models: {}, {} \pm {}'.format(model, np.mean(dists), np.std(dists)))