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


class LR_MNIST(nn.Module):
    # logistic regression for mnist 
    def __init__(self, n_classes):
        super(LR_MNIST, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=784, out_features=n_classes),
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


@torch.no_grad()
def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    model.eval()
    for batch in data_loader:
        X, y_true = batch[0], batch[1]
        X = X.to(device)
        y_true = y_true.to(device)

        _, y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def train_step(X, y_true, model, criterion, optimizer, device, step=True):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    optimizer.zero_grad()
    
    X = X.to(device)
    y_true = y_true.to(device)

    # Forward pass
    y_hat, _ = model(X) 
    loss = criterion(y_hat, y_true) 
    running_loss += loss.item() * X.size(0)

    # Backward pass
    loss.backward()
    model_grad = get_grad_list(model)
    if step:
        optimizer.step()
        
    # epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, running_loss, model_grad


def get_grad_list(model):
    grads = []
    for param in model.parameters():
        if param.requires_grad:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


def forge(FORGE_TYPE, 
          model_ori, model_ori_grad, model_forge, 
          X_cand, y_cand, criterion, 
          BATCH_SIZE, LEARNING_RATE, device):

    ind = np.random.choice(X_cand.shape[0], BATCH_SIZE, replace=False)
    X, y = X_cand[ind], y_cand[ind]
    
    if FORGE_TYPE == 'model':
        # min \| w_i - \tilde{w_{i-1}} + lr * grad(\tilde{w_{i-1}}, forged_batch) \|_2^2

        model = deepcopy(model_forge)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        model, _, _, _ = train_step(X, y, model, criterion, optimizer, device)
        dist = calc_model_dist(model, model_ori)

    elif FORGE_TYPE == 'grad':
        # min \|grad(w_{i-1}, ori_batch) - grad(\tilde{w_{i-1}}, forged_batch) \|_2^2

        model = deepcopy(model_forge)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        _, _, _, model_grad = train_step(X, y, model, criterion, optimizer, device, step=False)
        dist = torch.sum((model_grad - model_ori_grad) ** 2).item()

    elif FORGE_TYPE == 'paper':
        # min \|grad(w_{i-1}, ori_batch) - grad(w_{i-1}, forged_batch) \|_2^2
        
        model = deepcopy(model_ori)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        _, _, _, model_grad = train_step(X, y, model, criterion, optimizer, device, step=False)
        dist = torch.sum((model_grad - model_ori_grad) ** 2).item()

    return ind, dist


def forge_M(FORGE_TYPE, 
            model_ori, model_ori_grad, model_forge,
            X_cand, y_cand, criterion, 
            BATCH_SIZE, LEARNING_RATE, M, device):

    inds, dists = [], []
    for _ in range(M):
        ind, dist = forge(FORGE_TYPE, 
            model_ori, model_ori_grad, model_forge, 
            X_cand, y_cand, criterion, 
            BATCH_SIZE, LEARNING_RATE, 
            device
        )
        dists.append(dist)
        inds.append(ind)
    best_m = np.argmax(-np.array(dists))
    return inds[best_m], dists[best_m]


def flatten(inds):
    x = []
    for ind in inds:
        if not len(x):
            x = ind 
        else:
            x += ind 
    return x


@torch.no_grad()
def calc_model_dist(model1, model2):
    dist = 0.0
    P1 = model1.parameters()
    P2 = model2.parameters()
    for p1, p2 in zip(P1, P2):
        # p1_np, p2_np = p1.data.cpu().numpy(), p2.data.cpu().numpy()
        dist += torch.sum((p1 - p2) ** 2).item()
    return dist


def main(args):
    device = 'cuda:{}'.format(args.gpu)

    # parameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LOG_EVERY = args.log_every
    VALID_EVERY = args.valid_every

    FORGE = args.forge
    FORGE_TYPE = args.forge_type
    DELETED_BATCH_SIZE = args.n
    TRAIN_INDS = list(range(args.N_train))
    DELETE_IND = args.delete
    assert DELETE_IND in TRAIN_INDS
    M = args.M

    MODEL = args.model
    DATASET = args.dataset
    IMG_SIZE = args.img_size
    N_CLASSES = 10 if args.dataset in ['mnist'] else None

    data_folder = os.path.join('/tmp2', DATASET)
    os.makedirs(data_folder, exist_ok=True)
    tb_folder = "./tensorboard/{}-{}-type{}-Ntrain{}-del{}-n{}-M{}".format(
        DATASET, MODEL, FORGE_TYPE, len(TRAIN_INDS), DELETE_IND, DELETED_BATCH_SIZE, M
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
    remain_ind = list(set(TRAIN_INDS) - set([DELETE_IND]))
    deleted_dataset = Subset(full_dataset, remain_ind)

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
    model_forge = deepcopy(model_ori)
    optimizer_forge = torch.optim.SGD(model_forge.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model_ori.train()
    model_forge.train()

    print("Begin training: {} iterations per epoch; {} epochs total".format(len(train_loader), N_EPOCHS))
    train_losses = []
    train_inds = []
    forge_losses = []
    forge_inds = []
    model_dists = []

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
                # randomly select n data
                deleted_loader = DataLoader(dataset=deleted_dataset, 
                                            batch_size=DELETED_BATCH_SIZE, 
                                            shuffle=True,
                                            drop_last=True)
                for X_cand, y_cand, ind_cand in deleted_loader:
                    ind_cand = ind_cand.numpy()
                    break
                
                # forge M times using n data
                forge_ind, forge_eps = forge_M(
                    FORGE_TYPE, 
                    model_ori, model_ori_grad, model_forge, 
                    X_cand, y_cand, criterion, 
                    BATCH_SIZE, LEARNING_RATE, M, 
                    device
                )
                
                # update forged model
                X_forge, y_forge = X_cand[forge_ind], y_cand[forge_ind]
                model_forge, optimizer_forge, forge_loss, _ = train_step(
                    X_forge, y_forge, model_forge, criterion, optimizer_forge, device
                )
                
                # logging
                forge_losses.append(forge_loss / len(train_loader.dataset))
                forge_inds.append([ind_cand[_i] for _i in forge_ind])
                model_dist = calc_model_dist(model_ori, model_forge)
                model_dists.append(model_dist)
            else:
                forge_loss = 0
                model_dist = 0

            # logging 
            if iterations % LOG_EVERY == 0:
                writer.add_scalar('Loss/Train', train_loss, iterations)
                writer.add_scalar('Loss/Forge', forge_loss, iterations)
                writer.add_scalar('Loss/Model-Dist', model_dist, iterations)
                if FORGE_TYPE == 'paper':
                    writer.add_scalar('Loss/Forge-EPS', forge_eps, iterations)

            # validation
            if iterations % VALID_EVERY == 0:
                model_ori.eval()
                model_forge.eval()
                train_acc = get_accuracy(model_ori, train_loader, device=device)
                valid_acc = get_accuracy(model_ori, valid_loader, device=device)
                forge_acc = get_accuracy(model_forge, valid_loader, device=device)
                model_ori.train()
                model_forge.train()
                
                writer.add_scalar('Valid/Train-Acc', train_acc, iterations)
                writer.add_scalar('Valid/Valid-Acc', valid_acc, iterations)
                writer.add_scalar('Valid/Forge-Acc', forge_acc, iterations)

            iterations += 1

    # save files 
    torch.save({'train': train_inds, 'forge': forge_inds}, '{}/inds.pkl'.format(tb_folder))
    torch.save({'ori_state_dict': model_ori.state_dict(),
                'forge_state_dict': model_forge.state_dict()}, 
               '{}/model-{}epochs.pkl'.format(tb_folder, N_EPOCHS))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist'], help="dataset name")
    parser.add_argument("--model", type=str, default='LeNet5', choices=['LeNet5', 'LR'], help="network")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    
    parser.add_argument("--log_every", type=int, default=10, help="interval logging")
    parser.add_argument("--valid_every", type=int, default=50, help="interval validation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")

    # forging parameters 
    parser.add_argument("--forge", action="store_true", help="do forging")
    parser.add_argument("--forge_type", type=str, choices=['paper', 'grad', 'model'], help="forging loss type")
    parser.add_argument("--n", type=int, default=800, help="random n samples")
    parser.add_argument("--M", type=int, default=200, help="random M minibatches")
    parser.add_argument("--N_train", type=int, default=3200, help="first N training samples")
    parser.add_argument("--delete", type=int, default=0, help="delete sample ind")

    args = parser.parse_args()
    print('-'*50)
    print(args)

    main(args)
