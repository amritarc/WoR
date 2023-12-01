import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import entropy
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms

np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

from MI_forge_onesample import (
    LR_MNIST, LeNet5,
    dataset_with_indices, 
    get_accuracy,
    train_step,
    get_grad_list,
    flatten,
) 


def main(DATASET, MODEL, path, device):
    assert DATASET == 'mnist'
    data_folder = os.path.join('/tmp2', DATASET)

    IMG_SIZE = 32
    transform_fn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    valid_dataset = datasets.MNIST(root=data_folder, 
                                train=False, 
                                transform=transform_fn,
                                download=True)
    valid_loader = DataLoader(dataset=valid_dataset, 
                            batch_size=100, 
                            shuffle=False)
    
    N_CLASSES = 10
    if MODEL == 'LeNet5':
        model = LeNet5(N_CLASSES).to(device)
    elif MODEL == 'LR':
        model = LR_MNIST(N_CLASSES).to(device)
    
    # load model
    checkpoint = torch.load(path, map_location='cpu')
    if 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_ori_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_ori_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    valid_acc = get_accuracy(model, valid_loader, device=device)
    # print('valid_acc:', valid_acc)
    return valid_acc


if __name__ == '__main__':
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu))

    dataset = 'mnist'
    Ntrain = 60000
    # num_del = 10
    n_epochs = 20

    dic = defaultdict(list) 

    model = 'LeNet5'

    # for num_del in [10, 100]:
    #     EXP_PATH = '{}-{}-Ntrain{}-K5-M200'.format(
    #         dataset, model, Ntrain
    #     )

    #     for index in range(1, 11):
    #         path = os.path.join(
    #             './',
    #             'tensorboard_delete_{}'.format(num_del),
    #             EXP_PATH,
    #             'recovered-{}'.format(n_epochs),
    #             'del-{}'.format(index),
    #             'recovered.pkl'
    #         )
            
    #         valid_acc = main(dataset, model, path, device).cpu().item()
    #         dic[(model, num_del)].append(valid_acc)

    #     print('valid_acc (forged) - model {}, lambda={}: {} \pm {}'.format(
    #         model, 
    #         num_del, 
    #         np.mean(dic[(model, num_del)]), 
    #         np.std(dic[(model, num_del)])
    #     ))
    
    for seed in [1,2,3,4,5]:
        EXP_PATH = '{}-{}-Ntrain{}-seed{}'.format(
            dataset, model, Ntrain, seed
        )

        for index in range(1, 11):
            path = os.path.join(
                'tensorboard_randomseed',
                EXP_PATH,
                'model_ori_state_dict-{}.pkl'.format(n_epochs)
            )
            
            valid_acc = main(dataset, model, path, device).cpu().item()
            dic[(model)].append(valid_acc)

    print('valid_acc (random seed) - model {}: {} \pm {}'.format(
        model, 
        np.mean(dic[(model)]), 
        np.std(dic[(model)])
    ))
