import os 
import argparse
from tqdm import tqdm 

import numpy as np 
from scipy import stats
import torch 

import pandas as pd
from pandas.api.types import CategoricalDtype
from matplotlib import pyplot as plt
import seaborn as sns

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def parse_file(output_file):
    print('-'*30)
    print(output_file)

    Ndel = int(output_file.split('delete_')[1].split('-')[0])
    Ntrain = int(output_file.split('Ntrain')[1].split('-')[0])
    batchsize = 100
    epochs = 20

    l1_dist = torch.load(output_file, map_location='cpu')["l1_dist"]
    l1_dist = np.array(l1_dist)
    print("L1 dist: {:.4f} \pm {:.4f}".format(np.mean(l1_dist), np.std(l1_dist)))

    # generate freq_SGD by simulating SGD
    # print("simulating SGD")
    # UNIFORM = np.ones(Ntrain - Ndel) / (Ntrain - Ndel)
    # l1_dist_simulatedSGD = []
    # for _ in range(10):
    #     freq_SGD = np.array([0.0 for _ in range(Ntrain - Ndel)])
    #     for _ in range(int(Ntrain / batchsize) * epochs):
    #         minibatch = np.random.choice(Ntrain - Ndel, size=batchsize, replace=False)
    #         for ind in minibatch:
    #             freq_SGD[ind] += 1.0
    #     freq_SGD /= sum(freq_SGD)
    #     freq_SGD.sort()
    #     l1_dist_simulatedSGD.append(np.linalg.norm(freq_SGD - UNIFORM, ord=1))
    # print("Simulated SGD: {:.6f} \pm {:.6f}".format(np.mean(l1_dist_simulatedSGD), np.std(l1_dist_simulatedSGD)))


def plot():
    print("plotting batch frequency l1 dist from uniform")
    df_dic = {
        "algorithm": ["SGD"] * len(l1_dist_simulatedSGD),
        "L1 distance to uniform": l1_dist_simulatedSGD
    }
    for K in K_RANGE:
        df_dic["algorithm"] += ["forged (K={})".format(K)] * len(l1dist_K[K])
        df_dic["L1 distance to uniform"] += l1dist_K[K]

    df = pd.DataFrame(data=df_dic)
    plt.figure(figsize=(8,5))
    sns.violinplot(y=df["algorithm"], x=df["L1 distance to uniform"])
    plt.tight_layout()
    plt.savefig("L1dist_from_unif_allK-mnist-LeNet5-Ntrain{}-del{}-M{}.png".format(
        Ntrain, Ndel, M
    ), dpi=300)


if __name__ == '__main__':
    for output_file in sorted(os.listdir('./')):
        if output_file.endswith('.pkl'):
            parse_file(output_file)
