import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import entropy
import argparse

import sys
sys.path.append('../../Forgeability')

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms

from MI_forge_onesample import (
    LR_MNIST, LeNet5,
    dataset_with_indices, 
    get_accuracy,
    train_step,
    get_grad_list,
    flatten,
)


def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True, end="; ")
        
        if keyword is not None:
            keyword_parameters = [p for name, p in net.named_parameters() if p.requires_grad and keyword in name]
            params = sum([np.prod(p.size()) for p in keyword_parameters])
            print("{} Parameters: {:.6f}M".format(
                keyword, params / 1e6), flush=True, end="; ")
        
        print(" ")


def main(args):
    device = 'cuda:{}'.format(args.gpu)

    # parameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LOG_EVERY = args.log_every
    VALID_EVERY = args.valid_every
    SAVE_EVERY = args.save_every

    N_TRAIN = args.N_train
    TRAIN_INDS = list(range(N_TRAIN))

    MODEL = args.model
    DATASET = args.dataset
    IMG_SIZE = args.img_size
    N_CLASSES = 10 if args.dataset in ['mnist', 'cifar10'] else None

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    data_folder = os.path.join('/tmp2', DATASET)
    os.makedirs(data_folder, exist_ok=True)
    tb_folder = "./tensorboard/{}-{}-Ntrain{}-seed{}".format(
        DATASET, MODEL, N_TRAIN, SEED)
    os.makedirs(tb_folder, exist_ok=True)
    # clear tensorboard
    for tb_file in os.listdir(tb_folder):
        if tb_file.startswith('events'):
            os.remove(os.path.join(tb_folder, tb_file))
            print('Old tb file removed')
    writer = SummaryWriter(tb_folder)

    if args.dataset == 'mnist':
        transform_fn = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                        transforms.ToTensor()])

        MNISTWithIndices = dataset_with_indices(datasets.MNIST)
        full_dataset = MNISTWithIndices(root=data_folder, 
                                    train=True, 
                                    transform=transform_fn,
                                    download=True)
        valid_dataset = datasets.MNIST(root=data_folder, 
                                    train=False, 
                                    transform=transform_fn,
                                    download=True)

        train_dataset = Subset(full_dataset, TRAIN_INDS)
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)

        if MODEL == 'LeNet5':
            model_ori = LeNet5(N_CLASSES).to(device)
        elif MODEL == 'LR':
            model_ori = LR_MNIST(N_CLASSES).to(device)
    
    # save initial model
    torch.save({
            "epoch": 0, 
            "model_ori_state_dict": model_ori.state_dict()
        }, '{}/model_ori_state_dict_initialized.pkl'.format(tb_folder))

    optimizer_ori = torch.optim.SGD(model_ori.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()
    model_ori.train()

    print("Begin training: {} iterations per epoch; {} epochs total".format(len(train_loader), N_EPOCHS))
    train_losses = []
    train_inds = []

    # Train model
    iterations = 0
    for epoch in tqdm(range(N_EPOCHS)):
        for X, y_true, train_ind in train_loader:

            # training
            model_ori, optimizer_ori, train_loss, model_ori_grad = train_step(
                X, y_true, model_ori, criterion, optimizer_ori, device
            )
            train_losses.append(train_loss / len(train_loader.dataset))
            train_inds.append(train_ind.numpy())

            # logging 
            if iterations % LOG_EVERY == 0:
                writer.add_scalar('Loss/Train', train_loss, iterations)

            # validation
            if iterations % VALID_EVERY == 0:
                model_ori.eval()
                train_acc = get_accuracy(model_ori, train_loader, device=device)  # 10min on CIFAR
                valid_acc = get_accuracy(model_ori, valid_loader, device=device)
                model_ori.train()
                
                writer.add_scalar('Valid/Train-Acc', train_acc, iterations)
                writer.add_scalar('Valid/Valid-Acc', valid_acc, iterations)

            iterations += 1

        # save files 
        if (epoch+1) % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch+1, 
                "model_ori_state_dict": model_ori.state_dict()
            }, '{}/model_ori_state_dict-{}.pkl'.format(tb_folder, epoch+1))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist', 'cifar10'], help="dataset name")

    MODELS = 'LR, LeNet5'
    parser.add_argument("--model", type=str, default='LeNet5', choices=MODELS.split(', '), help="network")

    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD: learning rate")

    parser.add_argument("--log_every", type=int, default=10, help="interval logging")
    parser.add_argument("--valid_every", type=int, default=250, help="interval validation")
    parser.add_argument("--save_every", type=int, default=10, help="interval (epoch) saving model_ori ckpt")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")

    parser.add_argument("--N_train", type=int, default=60000, help="first N training samples")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    print('-'*50)
    print(args)

    main(args)
