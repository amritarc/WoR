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


# def estimate_thr(dataset, mean_in, std_in, mean_out, std_out):
#     # page 6, http://www.math.tau.ac.il/~turkel/notes/threshold.pdf
#     A = std_in ** 2 - std_out ** 2
#     B = 2 * (mean_in * std_out ** 2 - mean_out * std_in ** 2)
#     C = mean_out ** 2 * std_in ** 2 - mean_in ** 2 * std_out ** 2 + 4 * std_in ** 2 * std_out ** 2 * np.log(std_out / std_in)
#     # A T^2 + B T + C = 0

#     return thera_opt, in_is_left

def get_MI_thr(dataset, fix_variance):
    if dataset == 'mnist-10k':
        if fix_variance:
            return 0.016770097798849637
        else:
            return 2.7609440345292056
    if dataset == 'mnist-60k':
        if fix_variance:
            return -0.038821336711215126
        else:
            return 0.28327224348317065
    elif dataset == 'cifar10-50k':
        if fix_variance:
            return -0.02954835447712023
        else:
            return -0.06633275179702547
    else:
        raise NotImplementedError


def compute_MI_score(
    dataset, Ntrain, delete_ind, num_del, eval_split,
    model, model_ori, model_forge_path, forging_path, 
    score_files, keep_files, fix_variance, 
    device
):
    """
    Perform MI-LiRA for model on eval_split
        diff: from delete_ind * num_del to (delete_ind+1) * num_del
        train_diff: random points in train but not in diff
        test: random points in test
    """
    assert eval_split in ['diff', 'train_diff', 'test']

    # load dataset
    if dataset.startswith('mnist'):
        if model == 'LeNet5':
            IMG_SIZE = 32
        else:
            raise NotImplementedError

        transform_fn = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
        MNISTWithIndices = dataset_with_indices(datasets.MNIST)
        full_dataset = MNISTWithIndices(root=os.path.join('/tmp2', dataset), 
                                        train=True if eval_split != 'test' else False, 
                                        transform=transform_fn,
                                        download=True)
        sub_dataset = Subset(full_dataset, list(range(Ntrain))) if eval_split != 'test' else full_dataset
        dataloader = DataLoader(dataset=sub_dataset, 
                                batch_size=1, 
                                shuffle=False)
        N_CLASSES = 10
    else:
        raise NotImplementedError

    # load model
    checkpoint_forge = torch.load(model_forge_path, map_location='cpu')
    model_forge = LeNet5(N_CLASSES).to(device)
    model_forge.load_state_dict(checkpoint_forge['model_state_dict'])
    model_forge.eval()

    # load pre-computed shadow model scores
    score_shadow = [np.load(f) for f in score_files]  # List of 16 (Ntrain, 2) arrays
    keeps_shadow = [np.load(f)[:Ntrain] for f in keep_files]  # List of 16 (Ntrain, ) bool arrays
    assert len(keeps_shadow) == len(score_shadow)  # == 16

    logl_dic = {'original': [], 'forged': []}  # i-th element in each value is i-th logl; total len = num_del

    # which samples are used to compute MI: DIFF, Train_DIFF, Test
    deleted_indices = list(range(delete_ind * num_del, (delete_ind+1) * num_del))
    if eval_split == 'diff':
        indices_to_compute_MI = deleted_indices
    elif eval_split == 'train_diff':
        indices_to_compute_MI = np.random.choice(list(set(list(range(Ntrain))) - set(deleted_indices)), 5*num_del, replace=False)
    elif eval_split == 'test':
        indices_to_compute_MI = np.random.choice(len(sub_dataset), 5*num_del, replace=False)
    else:
        raise NotImplementedError

    for delete in indices_to_compute_MI:
        # compute confidence scores
        score_ori = compute_conf(model_ori, dataloader, delete, device)  # (2, ), 2 due to augmentation
        score_forge = compute_conf(model_forge, dataloader, delete, device)  # (2, )

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

        # # estimate best threshold 
        # thera_opt, in_is_left = estimate_thr(dataset, mean_in, std_in, mean_out, std_out)

    
        for sc_name, sc in zip(['original', 'forged'], [score_ori, score_forge]):
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            logl = (pr_in - pr_out).mean()
            logl_dic[sc_name].append(logl)
            
            # print('log-likelihood of sample {} in {} model (fix var {}): {:.2f}'.format(
            #     delete, sc_name, fix_variance, logl
            # ), end=', ')
            # print('inferring {}'.format('USED' if logl > 0 else 'NOT USED'))
    
    return logl_dic, [checkpoint_forge['model_dist']] * num_del


def main(dataset, model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, fix_variance, recompute, device):
    print('{}-del{}-epoch{}-M{}:\n'.format(dataset, num_del, n_epochs, M))
    print('eval_split:', eval_split)

    if dataset.endswith('-10k'):
        score_files = ["./exp/{}/experiment-{}_16/scores/0000000100.npy".format(dataset, i) for i in range(16)]
    else:
        score_files = ["./exp/{}/experiment-{}_16/scores/0000000020.npy".format(dataset, i) for i in range(16)]
    keep_files = ["./exp/{}/experiment-{}_16/keep.npy".format(dataset, i) for i in range(16)]

    dist_name = r"$\|\theta_*-\theta_{-i}\|_2^2$"
    logldiff_name = r"$\log\ \Lambda_i(\theta_*)-\log\ \Lambda_i(\theta_{-i})$"
    logldiv_name = r"$\frac{\log\ \Lambda_i(\theta_{-i})}{\log\ \Lambda_i(\theta_*)}-1$"
    df_dic = {
        "K": [],
        "logl_original": [],
        "logl_forged": []
    }
    if eval_split == 'diff':
        df_dic = {
            **df_dic, 
            "exp_id": [],
            dist_name: [],
            logldiff_name: [],
            logldiv_name: []
        }
        

    for splitK in K_RANGE:
        print("K={}".format(splitK))

        # get output path ready
        forging_path = "{}-{}-Ntrain{}-K{}-M{}".format(dataset.split('-')[0], model, Ntrain, splitK, M)
        output_file = './exp/{}/del-{}-epoch-{}/{}-split-{}.pkl'.format(dataset, num_del, n_epochs, forging_path, eval_split)
        os.makedirs('./exp/{}/del-{}-epoch-{}'.format(dataset, num_del, n_epochs), exist_ok=True)

        if recompute:
            model_ori_path = os.path.join(
                "../../Forgeability/tensorboard_delete_{}/".format(num_del),
                forging_path,
                "recovered-{}".format(n_epochs),
                "model_ori_state_dict.pkl"  # "forge.pkl"
            )
            checkpoint_ori = torch.load(model_ori_path, map_location='cpu')
            N_CLASSES = 10
            model_ori = LeNet5(N_CLASSES).to(device)
            model_ori.load_state_dict(checkpoint_ori['model_ori_state_dict'])
            model_ori.eval()

            logl_original = []
            logl_forged = []
            model_dists = []
            exp_id = []

            for delete_ind in tqdm(range(0, int(Ntrain // num_del))):  # int(Ntrain // num_del)
                model_forge_path = os.path.join(
                    "../../Forgeability/tensorboard_delete_{}/".format(num_del),
                    forging_path,
                    "recovered-{}/del-{}".format(n_epochs, delete_ind),
                    "recovered.pkl"
                )

                if not os.path.exists(model_forge_path):
                    continue

                # print('delete {}:'.format(delete))
                logl_dic, model_dist = compute_MI_score(
                    dataset, Ntrain, delete_ind, num_del, eval_split,
                    model, model_ori, model_forge_path, forging_path, 
                    score_files, keep_files, fix_variance, 
                    device
                )
                logl_original += logl_dic['original']
                logl_forged += logl_dic['forged']
                model_dists += model_dist
                exp_id += [delete_ind] * num_del
            
            # logl_original = np.array(logl_original)
            # logl_forged = np.array(logl_forged)
            # model_dists = np.array(model_dists)
        
            torch.save({
                'exp_id': exp_id,
                'num_del': num_del,
                'logl_original': logl_original,
                'logl_forged': logl_forged,
                'model_dists': model_dists
            }, output_file)
        
        else:
            dic = torch.load(output_file)
            logl_original = dic['logl_original']
            logl_forged = dic['logl_forged']
            model_dists = dic['model_dists']

            if num_del != 1:
                num_del = dic['num_del']
                exp_id = dic['exp_id']
            else:
                num_del = 1
                exp_id = list(range(0, int(Ntrain // num_del)))
        
        df_dic["K"] += [splitK] * len(logl_original)
        df_dic["logl_original"] += list(logl_original)
        df_dic["logl_forged"] += list(logl_forged)

        if eval_split == 'diff':
            df_dic["exp_id"] += exp_id
            df_dic[dist_name] += model_dists
            df_dic[logldiff_name] += list(np.array(logl_original) - np.array(logl_forged))
            df_dic[logldiv_name] += list(np.array(logl_forged) / np.array(logl_original) - 1.0)

        # print MI correctness statistics
        logl_original = np.array(logl_original) 
        logl_forged = np.array(logl_forged)
        thr = get_MI_thr(dataset, fix_variance)
        # MI_res = {  # T is used, F is unused, 1st is ori, 2nd is forged
        #     'TT': sum((logl_original >= thr) * (logl_forged >= thr)),
        #     'TF': sum((logl_original >= thr) * (logl_forged <  thr)),
        #     'FT': sum((logl_original <  thr) * (logl_forged >= thr)),
        #     'FF': sum((logl_original <  thr) * (logl_forged <  thr)),
        # }

    df = pd.DataFrame(data=df_dic)
    # df["K"] = df["K"].astype(CategoricalDtype(df["K"]))

    # evaluate MI prediction (hard)
    for splitK in K_RANGE:
        print("K={}".format(splitK))
        df_K = df[df["K"] == splitK]

        # MI_pred_on_DIFF = {'exist_wrong': 0, 'total': 0}
        # for exp_id in range(0, int(Ntrain // num_del)):
        #     df_exp_id = df_K[df_K['exp_id'] == exp_id]
        #     if len(df_exp_id) == 0:
        #         continue
        #     if sum((df_exp_id["logl_original"] >= thr) ^ (df_exp_id["logl_forged"] >= thr)) > 0:
        #         MI_pred_on_DIFF['exist_wrong'] += 1
        #     MI_pred_on_DIFF['total'] += 1

        #     MI_pred_on_DIFF['exist_wrong'] += sum((df_exp_id["logl_original"] >= thr) ^ (df_exp_id["logl_forged"] >= thr))
        #     MI_pred_on_DIFF['total'] += len(df_exp_id["logl_original"])
        # print('{} out of {} models have >= 1 different MI pred on subset DIFF'.format(MI_pred_on_DIFF['exist_wrong'], MI_pred_on_DIFF['total']))

        print('{} out of {} models have different MI pred'.format(
            sum((df_K["logl_original"] >= thr) ^ (df_K["logl_forged"] >= thr)),
            len(df_K)
        ))



    # evaluate MI outputs (soft)
    outlier_scale = 3  # default = 1.5

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[dist_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[dist_name] < high) & (df[dist_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[dist_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/model_dist-M{}.png'.format(dataset, num_del, n_epochs, M), dpi=300)

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[logldiff_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[logldiff_name] < high) & (df[logldiff_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[logldiff_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/logl_diff-M{}.png'.format(dataset, num_del, n_epochs, M), dpi=300)

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[logldiv_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[logldiv_name] < high) & (df[logldiv_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[logldiv_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/logl_div-M{}.png'.format(dataset, num_del, n_epochs, M), dpi=300)

    # quantiles = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
    # for col in [dist_name, logldiff_name, logldiv_name]:
    #     print('\ncol={}'.format(col))
    #     res = {}
    #     for splitK in K_RANGE:
    #         res[splitK] = np.percentile(df[df["K"] == splitK][col], quantiles)
    #     for i, q in enumerate(quantiles):
    #         for splitK in K_RANGE:
    #             if splitK == '10':
    #                 end = " \\\\\n"
    #             else:
    #                 end = " & "
    #             # print("{}".format(to_latex(res[splitK][i])), end=end)
    #             print("{}".format(res[splitK][i]), end=end)
    #     print('')

    print('-'*30)


if __name__ == "__main__":
    print('-'*30)

    gpu = 0
    device = torch.device('cuda:{}'.format(gpu))

    # dataset = 'mnist-10k'
    # model = 'LeNet5'
    # K_RANGE = ['5', '10']  # compare between hyperparams
    # Ntrain = 10000
    # M = 200
    # num_del = 1
    # n_epochs = 200

    dataset = 'mnist-60k'
    model = 'LeNet5'
    K_RANGE = ['5', '10']  # compare between hyperparams
    Ntrain = 60000
    M = 200
    num_del = 100
    n_epochs = 20

    # eval_split = 'train_diff'

    fix_variance = False
    recompute = False

    for eval_split in ['diff', 'train_diff', 'test']:
        main(dataset, model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, fix_variance, recompute, device)

