import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import entropy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

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
    calc_model_dist
)

# TODO: save model_forge in the middle


def main(args):
    device = 'cuda:{}'.format(args.gpu)

    # parameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LOG_EVERY = args.log_every

    DELETE_START, DELETE_END = args.delete.split(":")
    DELETE_START, DELETE_END = int(DELETE_START), int(DELETE_END)
    N_TRAIN = args.N_train
    TRAIN_INDS = list(range(N_TRAIN))
    NUM_DEL_EACH_EXP = args.num_del
    SPLITK = args.splitK
    M = args.M

    MODEL = args.model
    DATASET = args.dataset
    IMG_SIZE = args.img_size
    N_CLASSES = 10 if args.dataset in ['mnist'] else None

    data_folder = os.path.join('/tmp2', DATASET)
    os.makedirs(data_folder, exist_ok=True)
    exp_folder = "./tensorboard_delete_{}/{}-{}-Ntrain{}-K{}-M{}".format(
        NUM_DEL_EACH_EXP, DATASET, MODEL, N_TRAIN, SPLITK, M
    )
    assert os.path.exists(exp_folder)

    transform_fn = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                    transforms.ToTensor()])

    MNISTWithIndices = dataset_with_indices(datasets.MNIST)
    full_dataset = MNISTWithIndices(root=data_folder, 
                                train=True, 
                                transform=transform_fn,
                                download=True)

    train_dataset = Subset(full_dataset, TRAIN_INDS)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True)

    if MODEL == 'LeNet5':
        model_forge_init = LeNet5(N_CLASSES).to(device)
    elif MODEL == 'LR':
        model_forge_init = LR_MNIST(N_CLASSES).to(device)
    model_ori = deepcopy(model_forge_init)
    
    # load inds
    print('loading indices from {}'.format(os.path.join(exp_folder, "forge.pkl")))
    forge_checkpoint = torch.load(os.path.join(exp_folder, "forge.pkl"), map_location='cpu')
    print("model has been trained for {} epochs, use {} epochs".format(forge_checkpoint["epoch"] + 1, N_EPOCHS))

    # recover original model
    N_ITERATIONS = N_EPOCHS * len(train_loader)
    tb_folder_base = os.path.join(exp_folder, 'recovered-{}'.format(N_EPOCHS))
    os.makedirs(tb_folder_base, exist_ok=True)
    if os.path.exists(os.path.join(tb_folder_base, 'model_ori_state_dict.pkl')):
        print('load model_ori from {}/model_ori_state_dict.pkl'.format(tb_folder_base))
        model_ori.load_state_dict(torch.load(
            os.path.join(tb_folder_base, 'model_ori_state_dict.pkl'),
            map_location='cpu'
        )['model_ori_state_dict'])
    else:
        print('compute model_ori from saved inds')
        train_inds = forge_checkpoint["train_inds"]
        assert len(train_inds) >= N_ITERATIONS
        train_inds = train_inds[:N_ITERATIONS]
        ori_loader = DataLoader(dataset=train_dataset, batch_sampler=train_inds)

        optimizer_ori = torch.optim.SGD(model_ori.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        model_ori.train()

        iterations = 0
        for X, y_true, ori_ind in tqdm(ori_loader):
            assert all(ori_ind == torch.Tensor(train_inds[iterations]).long())
            model_ori, optimizer_ori, ori_loss, model_ori_grad = train_step(
                X, y_true, model_ori, criterion, optimizer_ori, device
            )
            iterations += 1
        
        torch.save({
                "epoch": N_EPOCHS,
                "model_ori_state_dict": model_ori.state_dict(),
            }, os.path.join(tb_folder_base, 'model_ori_state_dict.pkl'))

    # recover forged models
    for DELETE in tqdm(range(DELETE_START, DELETE_END)):
        tb_folder = os.path.join(tb_folder_base, 'del-{}'.format(DELETE))
        if os.path.exists(os.path.join(tb_folder, 'recovered.pkl')):
            continue
        else:  
            os.makedirs(tb_folder, exist_ok=True)

            # clear tensorboard
            for tb_file in os.listdir(tb_folder):
                if tb_file.startswith('events'):
                    os.remove(os.path.join(tb_folder, tb_file))
                    print('Old tb file removed')
            # writer = SummaryWriter(tb_folder)

        # print("Begin training (delete {}): {} iterations per epoch; {} epochs total".format(
        #     DELETE, len(train_loader), N_EPOCHS
        # ))

        forged_inds = forge_checkpoint["forge_inds"][DELETE]
        assert len(forged_inds) >= N_ITERATIONS
        forged_inds = forged_inds[:N_ITERATIONS]
        forge_loader = DataLoader(dataset=train_dataset, batch_sampler=forged_inds)

        model_forge = deepcopy(model_forge_init)
        optimizer_forge = torch.optim.SGD(model_forge.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        model_forge.train()

        iterations = 0
        forge_losses = []
        for X, y_true, forge_ind in (forge_loader):
            assert all(forge_ind == torch.Tensor(forged_inds[iterations]).long())

            # training
            model_forge, optimizer_forge, forge_loss, model_forge_grad = train_step(
                X, y_true, model_forge, criterion, optimizer_forge, device
            )

            # logging
            forge_losses.append(forge_loss / len(train_loader.dataset))
            # if iterations % LOG_EVERY == 0:
            #     writer.add_scalar('Loss/Forge', forge_loss, iterations)
            
            iterations += 1

        # compare distance to original model
        # model_ori = deepcopy(model_forge)
        # model_ori.load_state_dict(forge_checkpoint["model_ori_state_dict"])
        model_dist = calc_model_dist(model_ori, model_forge)

        # save files
        # print("model distance: {}".format(model_dist)) 
        torch.save({
            "epoch": N_EPOCHS,
            "model_state_dict": model_forge.state_dict(),
            "forge_losses": forge_losses,
            "model_dist": model_dist
        }, '{}/recovered.pkl'.format(tb_folder))

        # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist'], help="dataset name")
    parser.add_argument("--model", type=str, default='LeNet5', choices=['LeNet5', 'LR'], help="network")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    
    parser.add_argument("--log_every", type=int, default=10, help="interval logging")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")

    # forging parameters 
    parser.add_argument("--M", type=int, default=200, help="random M minibatches")
    parser.add_argument("--N_train", type=int, default=60000, help="first N training samples")
    parser.add_argument("--num_del", type=int, default=1, help="number of samples to delete for each exp")
    parser.add_argument("--splitK", type=int, default=5, help="split training data into K splits for each iteration")
    parser.add_argument("--delete", type=str, default="0:1", help="delete sample inds, start:end")

    args = parser.parse_args()
    print('-'*50)
    print(args)

    main(args)
