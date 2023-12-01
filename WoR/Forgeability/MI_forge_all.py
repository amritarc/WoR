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
)

# TODO: save model_ori in the middle


def random_splitK(TRAIN_INDS, NUM_DEL_EACH_EXP, K):
    # split [0,1,...,length-1] into K random splits, keep [i*num_del:(i+1)*num_del] in same split
    assert len(TRAIN_INDS) % (NUM_DEL_EACH_EXP * K) == 0
    _inds_splitK = list(range(int(len(TRAIN_INDS) / NUM_DEL_EACH_EXP)))
    np.random.shuffle(_inds_splitK)
    size_each_splitK = int(np.ceil(len(_inds_splitK) // K))
    inds_splitK = {}
    for k in range(K):
        inds_splitK[k] = {
            "del": _inds_splitK[k * size_each_splitK : (k+1) * size_each_splitK],
            "rem": _inds_splitK[:k * size_each_splitK] + _inds_splitK[(k+1) * size_each_splitK:],
        }
    return inds_splitK


def forge(model_ori, model_ori_grad, 
          X_cand, y_cand, criterion, 
          BATCH_SIZE, LEARNING_RATE, device):

    ind = np.random.choice(X_cand.shape[0], BATCH_SIZE, replace=False)
    X, y = X_cand[ind], y_cand[ind]
    
    # min \|grad(w_{i-1}, ori_batch) - grad(w_{i-1}, forged_batch) \|_2^2
    
    model = deepcopy(model_ori)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    _, _, _, model_grad = train_step(X, y, model, criterion, optimizer, device, step=False)
    dist = torch.sum((model_grad - model_ori_grad) ** 2).item()

    return ind, dist


def forge_M(model_ori, model_ori_grad,
            X_cand, y_cand, criterion, 
            BATCH_SIZE, LEARNING_RATE, M, device):

    inds, dists = [], []
    for _ in range(M):
        ind, dist = forge(
            model_ori, model_ori_grad, 
            X_cand, y_cand, criterion, 
            BATCH_SIZE, LEARNING_RATE, 
            device
        )
        dists.append(dist)
        inds.append(ind)
    best_m = np.argmax(-np.array(dists))
    return inds[best_m], dists[best_m]


def main(args):
    device = 'cuda:{}'.format(args.gpu)

    # parameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LOG_EVERY = args.log_every
    VALID_EVERY = args.valid_every
    SAVE_EVERY = args.save_every

    FORGE = args.forge
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
    if not FORGE:
        tb_folder = "./tensorboard_randomseed/{}-{}-Ntrain{}-seed{}".format(
            DATASET, MODEL, N_TRAIN, args.seed)
    else:
        tb_folder = "./tensorboard_delete_{}/{}-{}-Ntrain{}-K{}-M{}".format(
            NUM_DEL_EACH_EXP, DATASET, MODEL, N_TRAIN, SPLITK, M
        )
    os.makedirs(tb_folder, exist_ok=True)
    # clear tensorboard
    for tb_file in os.listdir(tb_folder):
        if tb_file.startswith('events'):
            os.remove(os.path.join(tb_folder, tb_file))
            print('Old tb file removed')
    writer = SummaryWriter(tb_folder)

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
        
    optimizer_ori = torch.optim.SGD(model_ori.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model_ori.train()

    print("Begin training: {} iterations per epoch; {} epochs total".format(len(train_loader), N_EPOCHS))
    train_losses = []
    train_inds = []
    forge_inds = [[] for _ in range(int(N_TRAIN // NUM_DEL_EACH_EXP))]
    forge_epses = [[] for _ in range(int(N_TRAIN // NUM_DEL_EACH_EXP))]

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

            # forging
            if FORGE:
                # split data into K splits
                inds_splitK = random_splitK(TRAIN_INDS, NUM_DEL_EACH_EXP, SPLITK)

                forge_eps = 0
                for k in range(SPLITK):
                    remaining_inds = []
                    for _rem_ind in inds_splitK[k]["rem"]:
                        remaining_inds += list(range(_rem_ind * NUM_DEL_EACH_EXP, (_rem_ind+1) * NUM_DEL_EACH_EXP))

                    deleted_loader = DataLoader(dataset=Subset(full_dataset, remaining_inds), 
                                                batch_size=4*M, 
                                                shuffle=False,
                                                drop_last=False)
                    for X_cand, y_cand, ind_cand in deleted_loader:
                        ind_cand = ind_cand.numpy()
                        break
                    
                    # forge M times using n data
                    forge_ind, forge_eps_k = forge_M(
                        model_ori, model_ori_grad, 
                        X_cand, y_cand, criterion, 
                        BATCH_SIZE, LEARNING_RATE, M, 
                        device
                    )
                    forge_eps = max(forge_eps_k, forge_eps)
                    
                    # logging
                    forged_batch = [ind_cand[_i] for _i in forge_ind]
                    ori_batch = list(train_ind.numpy())
                    for _del_ind in inds_splitK[k]["del"]:
                        all_del_inds = set(list(range(_del_ind * NUM_DEL_EACH_EXP, (_del_ind+1) * NUM_DEL_EACH_EXP)))
                        if len(all_del_inds & set(ori_batch)) > 0:
                            # if these inds intersect with original batch, use forged batch
                            forge_inds[_del_ind].append(forged_batch)
                            forge_epses[_del_ind].append(forge_eps_k)
                        else:
                            # otherwise use original batch
                            forge_inds[_del_ind].append(ori_batch)
                            forge_epses[_del_ind].append(0)
                        
                        # double check: deleted inds not appear
                        assert not (set(forge_inds[_del_ind][-1]) & all_del_inds)
                    
            else:
                pass

            # logging 
            if iterations % LOG_EVERY == 0:
                writer.add_scalar('Loss/Train', train_loss, iterations)
                if FORGE:
                    writer.add_scalar('Loss/Forge-EPS', forge_eps, iterations)

            # validation
            if iterations % VALID_EVERY == 0:
                model_ori.eval()
                train_acc = get_accuracy(model_ori, train_loader, device=device)
                valid_acc = get_accuracy(model_ori, valid_loader, device=device)
                model_ori.train()
                
                writer.add_scalar('Valid/Train-Acc', train_acc, iterations)
                writer.add_scalar('Valid/Valid-Acc', valid_acc, iterations)

            iterations += 1

        # save files 
        if FORGE:
            torch.save({
                "epoch": epoch+1, 
                "train_losses": train_losses,
                "train_inds": train_inds,
                "train_flips": train_flips,
                "forge_inds": forge_inds,
                "forge_epses": forge_epses,
                "forge_flips": forge_flips,
                "model_ori_state_dict": model_ori.state_dict()
            }, '{}/forge.pkl'.format(tb_folder))
        if (epoch+1) % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch+1, 
                "model_ori_state_dict": model_ori.state_dict()
            }, '{}/model_ori_state_dict-{}.pkl'.format(tb_folder, epoch+1))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist'], help="dataset name")
    parser.add_argument("--model", type=str, default='LeNet5', choices=['LeNet5', 'LR'], help="network")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
    
    parser.add_argument("--log_every", type=int, default=10, help="interval logging")
    parser.add_argument("--valid_every", type=int, default=50, help="interval validation")
    parser.add_argument("--save_every", type=int, default=10, help="interval (epoch) saving model_ori ckpt")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # forging parameters 
    parser.add_argument("--forge", action="store_true", help="do forging")
    parser.add_argument("--M", type=int, default=200, help="random M minibatches")
    parser.add_argument("--N_train", type=int, default=60000, help="first N training samples")
    parser.add_argument("--num_del", type=int, default=1, help="number of samples to delete for each exp")
    parser.add_argument("--splitK", type=int, default=10, help="split training data into K splits for each iteration")

    args = parser.parse_args()
    print('-'*50)
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)
