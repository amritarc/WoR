import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 100
N_EPOCHS = 100
LOG_EVERY = 10
VALID_EVERY = 100

FORGE = True
DELETED_BATCH_SIZE = 500
TRAIN_INDS = list(range(5000))
DELETE_INDS = list(range(50))
M = 200

IMG_SIZE = 32
N_CLASSES = 10

data_folder = "/tmp2/mnist"
os.makedirs(data_folder, exist_ok=True)
tb_folder = "./tb_forgeability/MNIST-5000-50-M200"
os.makedirs(tb_folder, exist_ok=True)
writer = SummaryWriter(tb_folder)


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


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
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


def train_step(X, y_true, model, criterion, optimizer, device):
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
    optimizer.step()
        
    # epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, running_loss


def forge(model_ori, model_forge, X_cand, y_cand, criterion, device):
    ind = np.random.choice(X_cand.shape[0], BATCH_SIZE, replace=False)
    X, y = X_cand[ind], y_cand[ind]
    model = deepcopy(model_forge)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    model, _, _ = train_step(X, y, model, criterion, optimizer, device)
    dist = calc_model_dist(model, model_ori)
    return ind, dist


def forge_M(model_ori, model_forge, X_cand, y_cand, criterion, M, device):
    inds, dists = [], []
    for _ in range(M):
        ind, dist = forge(model_ori, model_forge, X_cand, y_cand, criterion, device)
        dists.append(dist)
        inds.append(ind)
    best_m = np.argmax(-np.array(dists))
    return inds[best_m]


def stat_ind_freq(inds, bins):
    hist = np.zeros(len(bins))
    for ind in inds:
        y, _ = np.histogram(ind, bins=bins+[len(bins)])
        hist += y 
    # hist.sort()
    return hist / sum(hist)


def flatten(inds):
    x = []
    for ind in inds:
        if not len(x):
            x = ind 
        else:
            x += ind 
    return x


def calc_model_dist(model1, model2):
    dist = 0.0
    with torch.no_grad():
        P1 = model1.parameters()
        P2 = model2.parameters()
        for p1, p2 in zip(P1, P2):
            p1_np, p2_np = p1.data.cpu().numpy(), p2.data.cpu().numpy()
            dist += np.sum((p1_np - p2_np) ** 2)
    return dist


def training_loop(model_ori, model_forge, 
                  criterion, 
                  optimizer_ori, optimizer_forge, 
                  train_loader, deleted_dataset, deleted_batch_size, M,
                  epochs, device):
    '''
    Function defining the entire training loop
    '''
    print("Begin training: {} iterations per epoch; {} epochs total".format(len(train_loader), epochs))
    train_losses = []
    train_inds = []
    forge_losses = []
    forge_inds = []
    model_dists = []
    bins = list(range(len(train_dataset)))

    # Train model
    iterations = 0
    for epoch in range(0, epochs):
        for X, y_true, train_ind in (train_loader):
            # training
            model_ori, optimizer_ori, train_loss = train_step(X, y_true, 
                                                              model_ori, criterion, 
                                                              optimizer_ori, device)
            train_losses.append(train_loss / len(train_loader.dataset))
            train_inds.append(train_ind.numpy())

            # forging
            if FORGE:
                # randomly select n data
                deleted_loader = DataLoader(dataset=deleted_dataset, 
                                            batch_size=deleted_batch_size, 
                                            shuffle=True)
                for X_cand, y_cand, ind_cand in deleted_loader:
                    ind_cand = ind_cand.numpy()
                    break
                
                # forge M times using n data
                forge_ind = forge_M(model_ori, model_forge, X_cand, y_cand, criterion, M, device)
                
                # update forged model
                X_forge, y_forge = X_cand[forge_ind], y_cand[forge_ind]
                model_forge, optimizer_forge, forge_loss = train_step(X_forge, y_forge, 
                                                                model_forge, criterion, 
                                                                optimizer_forge, device)
                
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
                hist_train = stat_ind_freq(train_inds, bins)
                hist_forge = stat_ind_freq(forge_inds, bins)
                E_train = entropy(hist_train)
                E_forge = entropy(hist_forge)
                writer.add_scalar('Stat/Train-Inds-Entropy', E_train, iterations)
                writer.add_scalar('Stat/Forge-Inds-Entropy', E_forge, iterations)

                print("Iter {}; Inds-Entropy: forge {:.2f}; train {:.2f}".format(iterations, E_forge, E_train))

            # validation
            if iterations % VALID_EVERY == 0:
                train_acc = get_accuracy(model_ori, train_loader, device=device)
                valid_acc = get_accuracy(model_ori, valid_loader, device=device)
                forge_acc = get_accuracy(model_forge, valid_loader, device=device)
                
                writer.add_scalar('Valid/Train-Acc', train_acc, iterations)
                writer.add_scalar('Valid/Valid-Acc', valid_acc, iterations)
                writer.add_scalar('Valid/Forge-Acc', forge_acc, iterations)

            iterations += 1

    # save files 
    torch.save({'train': train_inds, 'forge': forge_inds}, '{}/inds.pkl'.format(tb_folder))

    return model_ori, optimizer_ori, (train_losses, )


if __name__ == '__main__':
    transforms = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])

    MNISTWithIndices = dataset_with_indices(datasets.MNIST)
    full_dataset = MNISTWithIndices(root=data_folder, 
                                train=True, 
                                transform=transforms,
                                download=True)
    valid_dataset = datasets.MNIST(root=data_folder, 
                                train=False, 
                                transform=transforms)

    train_dataset = Subset(full_dataset, TRAIN_INDS)
    remain_ind = list(set(TRAIN_INDS) - set(DELETE_INDS))
    deleted_dataset = Subset(full_dataset, remain_ind)

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    model_ori = LeNet5(N_CLASSES).to(DEVICE)
    optimizer_ori = torch.optim.SGD(model_ori.parameters(), lr=LEARNING_RATE)
    model_forge = deepcopy(model_ori)
    optimizer_forge = torch.optim.SGD(model_forge.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model_ori, optimizer_ori, _ = training_loop(model_ori, model_forge, 
                                                criterion, optimizer_ori, optimizer_forge, 
                                                train_loader, deleted_dataset, DELETED_BATCH_SIZE, M,
                                                N_EPOCHS, DEVICE)

    writer.close()
