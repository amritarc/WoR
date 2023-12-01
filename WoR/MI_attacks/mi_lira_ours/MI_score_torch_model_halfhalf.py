import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
import scipy
# from scipy.stats import entropy
import argparse
import sys
sys.path.append("../../Forgeability")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

from matplotlib import pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns

np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

from MI_forge_onesample import LeNet5, dataset_with_indices, flatten


def to_latex(value):
    sign = np.sign(value)
    value = abs(value)
    power = int(np.floor(np.log10(value)))
    value = value / (10 ** power)
    if power == 0:
        return r"${}{:.2f}$".format('-' if sign < 0 else '', value)
    else:
        return r"${}{:.2f}\times10^{}{}{}$".format('-' if sign < 0 else '', value, '{', power, '}')


@torch.no_grad()
def compute_conf(model, dataloader, delete, device):
    """compute confidence scores of model on outputs"""

    del_sample = dataloader.dataset[delete]
    x_del, y_del, i_del = del_sample 
    assert i_del == delete 

    phi = lambda p: torch.log(p / (1.0 - p))
    flip_fn = transforms.RandomHorizontalFlip(p=1.0)

    # get all confidences
    x, y, i = x_del, y_del, i_del

    # removed for loop as only need to look at x_del
    if len(x.shape) == 3:
        x = x.unsqueeze(1)  # BHW -> BCHW
    x = x.to(device)

    score = []
    for aug in [x, flip_fn(x)]:
        _, probs = model(aug)
        score.append(probs[0, y_del].item())  # 0 because bachsize is 1

    score = np.array(score)
    return score


def compute_MI_score(
    dataset, Ntrain, delete, 
    model, model_ori, model_forge, 
    score_files, keep_files, fix_variance, 
    device
):
    """
    Perform MI-LiRA for model on x_delete
    """
    # load dataset
    if dataset.startswith('mnist'):
        if model == 'LeNet5':
            IMG_SIZE = 32
        else:
            raise NotImplementedError

        transform_fn = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
        MNISTWithIndices = dataset_with_indices(datasets.MNIST)
        full_dataset = MNISTWithIndices(root=os.path.join('/tmp2', dataset), 
                                        train=True, 
                                        transform=transform_fn,
                                        download=True)
        train_dataset = Subset(full_dataset, list(range(Ntrain)))
        dataloader = DataLoader(dataset=train_dataset, 
                                batch_size=1, 
                                shuffle=False)
        N_CLASSES = 10
    else:
        raise NotImplementedError

    # compute confidence scores
    score_ori = compute_conf(model_ori, dataloader, delete, device)  # (2, ), 2 due to augmentation
    score_forge = compute_conf(model_forge, dataloader, delete, device)  # (2, )

    # load pre-computed shadow model scores
    score_shadow = [np.load(f) for f in score_files]  # List of 16 (Ntrain, 2) arrays
    keeps_shadow = [np.load(f)[:Ntrain] for f in keep_files]  # List of 16 (Ntrain, ) bool arrays
    assert len(keeps_shadow) == len(score_shadow)  # == 16

    # estimate conf_in and conf_out based on pre-computed scores and keep files
    conf_in = []
    conf_out = []
    for keep, score in zip(keeps_shadow, score_shadow):
        if keep[delete]:
            conf_in.append(score[delete])
        else:
            conf_out.append(score[delete])

    # estimate mean and std
    mean_in = np.median(conf_in, 0)
    mean_out = np.median(conf_out, 0)
    # print(mean_in.shape, mean_out.shape)

    if fix_variance:
        std_in = np.std(conf_in)
        std_out = np.std(conf_out)
    else:
        std_in = np.std(conf_in, 0)
        std_out = np.std(conf_out, 0)
    
    logl_dic = {}
    for sc_name, sc in zip(['original', 'forged'], [score_ori, score_forge]):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        logl = (pr_in - pr_out).mean()
        logl_dic[sc_name] = logl
        
        # print('log-likelihood of sample {} in {} model (fix var {}): {:.2f}'.format(
        #     delete, sc_name, fix_variance, logl
        # ), end=', ')
        # print('inferring {}'.format('USED' if logl > 0 else 'NOT USED'))
    
    return logl_dic


if __name__ == "__main__":
    gpu = 1
    device = torch.device('cuda:{}'.format(gpu))

    dataset = 'mnist-60k'
    model = 'LeNet5'
    alg = "typepaper"
    Ntrain, Nforge = 30000, 30000
    B, n, M = 64, 1600, 200
    LR_RANGE = [0.01, 0.02, 0.05]  # compare between hyperparams
    n_epochs = 10
    fix_variance = False
    recompute = False

    score_files = ["./exp/{}/experiment-{}_16/scores/0000000010.npy".format(dataset, i) for i in range(16)]
    keep_files = ["./exp/{}/experiment-{}_16/keep.npy".format(dataset, i) for i in range(16)]

    # dist_name = r"$\|\theta_*-\theta_{-i}\|_2^2$"
    logldiff_name = r"$\log\ \Lambda_i(\theta_*)-\log\ \Lambda_i(\theta_{-i})$"
    logldiv_name = r"$\frac{\log\ \Lambda_i(\theta_{-i})}{\log\ \Lambda_i(\theta_*)}-1$"
    df_dic = {
        "index": [],
        "LR": [],
        logldiff_name: [],
        logldiv_name: []
    }

    for LR in LR_RANGE:
        # get output path ready
        forging_path = "{}-{}-halfhalf-{}-Ntrain{}-Nforge{}-B{}-LR{}-n{}-M{}".format(
            dataset.split('-')[0], 
            model, alg, 
            Ntrain, Nforge, 
            B, LR, n, M)
        output_file = './exp/{}/{}.pkl'.format(dataset, forging_path)

        if recompute:
            checkpoint_path = os.path.join(
                "../../Forgeability/tensorboard/",
                forging_path,
                "model-{}epochs.pkl".format(n_epochs)
            )
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            N_CLASSES = 10

            model_ori = LeNet5(N_CLASSES).to(device)
            model_ori.load_state_dict(checkpoint['ori_state_dict'])
            model_ori.eval()

            model_forge = LeNet5(N_CLASSES).to(device)
            model_forge.load_state_dict(checkpoint['forge_state_dict'])
            model_forge.eval()

            logl_original = []
            logl_forged = []

            for delete in tqdm(range(0, Ntrain + Nforge)):
                logl_dic = compute_MI_score(
                    dataset, Ntrain + Nforge, delete, 
                    model, model_ori, model_forge, 
                    score_files, keep_files, fix_variance, 
                    device
                )
                logl_original.append(logl_dic['original'])
                logl_forged.append(logl_dic['forged'])
            
            # logl_original = np.array(logl_original)
            # logl_forged = np.array(logl_forged)
        
            torch.save({
                'logl_original': logl_original,
                'logl_forged': logl_forged
            }, output_file)
        
        else:
            dic = torch.load(output_file)
            logl_original = dic['logl_original']
            logl_forged = dic['logl_forged']
        
        df_dic["index"] += list(range(Ntrain + Nforge))
        df_dic["LR"] += [LR] * len(logl_original)
        df_dic[logldiff_name] += list(np.array(logl_original) - np.array(logl_forged))
        df_dic[logldiv_name] += list(np.array(logl_forged) / np.array(logl_original) - 1.0)

        # print MI correctness statistics
        logl_original = np.array(logl_original) 
        logl_forged = np.array(logl_forged)
        MI_res = {  # T is used, F is unused, 1st is ori, 2nd is forged
            'TT': sum((logl_original >= 0) * (logl_forged >= 0)),
            'TF': sum((logl_original >= 0) * (logl_forged <  0)),
            'FT': sum((logl_original <  0) * (logl_forged >= 0)),
            'FF': sum((logl_original <  0) * (logl_forged <  0)),
        }
        print('MI used/unused results:', MI_res)

    df = pd.DataFrame(data=df_dic)
    # df["LR"] = df["LR"].astype(CategoricalDtype(df["LR"]))

    quantiles = [0, 1, 25, 50, 75, 99, 100]
    for col in [logldiff_name, logldiv_name]:
        print('col={}'.format(col))
        res = {}
        for LR in LR_RANGE:
            res[LR] = np.percentile(df[df["LR"] == LR][col], quantiles)
        for i, q in enumerate(quantiles):
            for LR in LR_RANGE:
                if LR == LR_RANGE[-1]:
                    end = " \\\\\n"
                else:
                    end = " & "
                print("{}".format(to_latex(res[LR][i])), end=end)
        print('')

    plt.figure(figsize=(8,5))
    q1, q3 = np.percentile(df[logldiff_name], [25, 75])
    low, high = q1 - (q3 - q1) * 1.5, q3 + (q3 - q1) * 1.5
    df_tmp = df[(df[logldiff_name] < high) & (df[logldiff_name] > low)]
    sns.violinplot(y=df_tmp["LR"], x=df_tmp[logldiff_name])
    plt.savefig('./exp/{}/logl_diff.png'.format(dataset), dpi=400)

    plt.figure(figsize=(8,5))
    q1, q3 = np.percentile(df[logldiv_name], [25, 75])
    low, high = q1 - (q3 - q1) * 1.5, q3 + (q3 - q1) * 1.5
    df_tmp = df[(df[logldiv_name] < high) & (df[logldiv_name] > low)]
    sns.violinplot(y=df_tmp["LR"], x=df_tmp[logldiv_name])
    plt.savefig('./exp/{}/logl_div.png'.format(dataset), dpi=400)

    

