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

from MI_forge_onesample import dataset_with_indices, flatten


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
        # probs = F.softmax(logits, dim=1)
        return logits


def to_latex(value):
    sign = np.sign(value)
    value = abs(value)
    power = int(np.floor(np.log10(value)))
    value = value / (10 ** power)
    if power == 0:
        return r"${}{:.2f}$".format('-' if sign < 0 else '', value)
    else:
        return r"${}{:.2f}\times10^{}{}{}$".format('-' if sign < 0 else '', value, '{', power, '}')


def _log_value(probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))


def _entr_comp(probs):
    return np.sum(np.multiply(probs, _log_value(probs)),axis=1)


def _thre_setting(tr_values, te_values):
    value_list = np.concatenate((tr_values, te_values))
    thre, max_acc = 0, 0
    for value in value_list:
        tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
        te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
        acc = 0.5*(tr_ratio + te_ratio)
        if acc > max_acc:
            thre, max_acc = value, acc
    return thre


def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator


def _model_predictions(model, dataloader, device):
    return_outputs, return_labels = [], []
    for (inputs, labels) in tqdm(dataloader):
        return_labels.append(labels.numpy())
        outputs = model.forward(inputs.to(device)) 
        return_outputs.append(softmax_by_row(outputs.data.cpu().numpy()) )
    return_outputs = np.concatenate(return_outputs)
    return_labels = np.concatenate(return_labels)
    return (return_outputs, return_labels)


@torch.no_grad()
def compute_or_load_threshold(model_shadow, trainloader, testloader, device, outfile):
    if os.path.exists(outfile):
        print('loading threshold from:', outfile)
        thr = torch.load(outfile, map_location='cpu')['thr']

    else:
        print('computing threshold ...')
        _outputs, tr_labels = _model_predictions(model_shadow, trainloader, device)
        tr_values = _entr_comp(_outputs)

        _outputs, te_labels = _model_predictions(model_shadow, testloader, device)
        te_values = _entr_comp(_outputs)

        # compute threshold for each label
        thr = []
        for label in range(len(set(tr_labels))):
            thr.append(_thre_setting(tr_values[tr_labels == label], te_values[te_labels == label]))

        dic = {
            'thr': thr,
            'tr_values': tr_values,
            'te_values': te_values
        } 
        torch.save(dic, outfile)
    
    print('threshold is:', thr)
    return thr


@torch.no_grad()
def compute_MI_score(
    dataset, dataloader, Ntrain, delete_ind, num_del, eval_split,
    model, model_ori, model_forge_path, forging_path, device):

    """
    Perform MI-LiRA for model on eval_split
        diff: from delete_ind * num_del to (delete_ind+1) * num_del
        train_diff: random points in train but not in diff
        test: random points in test
    """
    assert eval_split in ['diff', 'train_diff', 'test']

    # load model
    checkpoint_forge = torch.load(model_forge_path, map_location='cpu')
    model_forge = deepcopy(model_ori).to(device)
    model_forge.load_state_dict(checkpoint_forge['model_state_dict'])
    model_forge.eval()

    xent_dic = {'original': [], 'forged': [], 'labels': []}  # i-th element in each value is i-th xent; total len = num_del

    # which samples are used to compute MI: DIFF, Train_DIFF, Test
    deleted_indices = list(range(delete_ind * num_del, (delete_ind+1) * num_del))
    if eval_split == 'diff':
        indices_to_compute_MI = deleted_indices
    elif eval_split == 'train_diff':
        indices_to_compute_MI = np.random.choice(list(set(list(range(Ntrain))) - set(deleted_indices)), 5*num_del, replace=False)
    elif eval_split == 'test':
        indices_to_compute_MI = np.random.choice(len(dataloader.dataset), 5*num_del, replace=False)
    else:
        raise NotImplementedError

    # gather all samples to evaluate MI scores
    x_all, _labels = [], []
    for delete in indices_to_compute_MI:
        del_sample = dataloader.dataset[delete]
        x_del, y_del, i_del = del_sample 
        assert i_del == delete 
        x_all.append(x_del)
        _labels.append(y_del)
    x_all = torch.stack(x_all).to(device)
    _labels = np.array(_labels)
    xent_dic['labels'] = list(_labels)
    
    for sc_name, model in zip(['original', 'forged'], [model_ori, model_forge]):
        _outputs = model.forward(x_all) 
        _outputs = softmax_by_row(_outputs.data.cpu().numpy())
        xent = _entr_comp(_outputs)
        xent_dic[sc_name] = list(xent)
    
    return xent_dic, [checkpoint_forge['model_dist']] * num_del


def main(dataset, model, shadow_model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, 
         device):
    print('{}-del{}-epoch{}-M{}:\n'.format(dataset, num_del, n_epochs, M))
    print('eval_split:', eval_split)

    N_CLASSES = 10 if dataset.startswith('mnist') else None

    dist_name = r"$\|\theta_*-\theta_{-i}\|_2^2$"
    xentdiff_name = r"$\Lambda_i(\theta_*)-\Lambda_i(\theta_{-i})$"
    xentdiv_name = r"$\frac{\Lambda_i(\theta_{-i})}{\Lambda_i(\theta_*)}-1$"
    df_dic = {
        "K": [],
        "xent_labels": [],
        "xent_original": [],
        "xent_forged": []
    }
    if eval_split == 'diff':
        df_dic = {
            **df_dic, 
            "exp_id": [],
            dist_name: [],
            xentdiff_name: [],
            xentdiv_name: []
        }

    for splitK in K_RANGE:
        print("K={}".format(splitK))

        # get output path ready
        forging_path = "{}-{}-Ntrain{}-K{}-M{}".format(dataset.split('-')[0], model, Ntrain, splitK, M)
        output_file = './exp-xent/{}/del-{}-epoch-{}/{}-split-{}.pkl'.format(dataset, num_del, n_epochs, forging_path, eval_split)
        os.makedirs('./exp-xent/{}/del-{}-epoch-{}'.format(dataset, num_del, n_epochs), exist_ok=True)

        if recompute:
            if dataset.startswith('mnist'):
                N_CLASSES = 10

                if model == 'LeNet5':
                    model_ori = LeNet5(N_CLASSES).to(device)
                    IMG_SIZE = 32
                if shadow_model == 'LeNet5':
                    model_shadow = LeNet5(N_CLASSES).to(device)

                data_folder = os.path.join('/tmp2', dataset.split('-')[0])
                transform_fn = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

                # add train/test loader to compute thr via shadow model
                trainloader = DataLoader(
                    dataset=datasets.MNIST(root=data_folder, train=True, transform=transform_fn, download=True),
                    batch_size=64, shuffle=False, drop_last=False)
                testloader = DataLoader(
                    dataset=datasets.MNIST(root=data_folder, train=False, transform=transform_fn, download=True),
                    batch_size=64, shuffle=False, drop_last=False)

                MNISTWithIndices = dataset_with_indices(datasets.MNIST)
                full_dataset = MNISTWithIndices(root=data_folder, 
                                            train=True if eval_split != 'test' else False, 
                                            transform=transform_fn,
                                            download=True)
                sub_dataset = Subset(full_dataset, list(range(Ntrain))) if eval_split != 'test' else full_dataset
                dataloader = DataLoader(dataset=sub_dataset, 
                                        batch_size=1, 
                                        shuffle=False)

            else:
                raise NotImplementedError

            model_ori_path = os.path.join(
                "../../Forgeability/tensorboard_delete_{}/".format(num_del),
                forging_path,
                "recovered-{}".format(n_epochs),
                "model_ori_state_dict.pkl"  # "forge.pkl"
            )
            checkpoint_ori = torch.load(model_ori_path, map_location='cpu')
            model_ori.load_state_dict(checkpoint_ori['model_ori_state_dict'])
            model_ori.eval()

            model_shadow_path = model_ori_path.replace(model, shadow_model)
            checkpoint_shadow = torch.load(model_shadow_path, map_location='cpu')
            model_shadow.load_state_dict(checkpoint_shadow['model_ori_state_dict'])
            model_shadow.eval()

            # compute threshold
            thr = compute_or_load_threshold(
                model_shadow, trainloader, testloader, device, 
                outfile='./exp-xent/{}/del-{}-epoch-{}/{}-thr.pkl'.format(dataset, num_del, n_epochs, forging_path)
            )

            xent_labels = []
            xent_original = []
            xent_forged = []
            model_dists = []
            exp_id = []

            for delete_ind in tqdm(range(0, int(Ntrain // num_del))):  # int(Ntrain // num_del)
                model_forge_path = os.path.join(
                    "/home/USERNAME/MI_project/Forgeability/tensorboard_delete_{}/".format(num_del),
                    forging_path,
                    "recovered-{}/del-{}".format(n_epochs, delete_ind),
                    "recovered.pkl"
                )

                if not os.path.exists(model_forge_path):
                    continue

                # print('delete {}:'.format(delete))
                xent_dic, model_dist = compute_MI_score(
                    dataset, dataloader, Ntrain, delete_ind, num_del, eval_split,
                    model, model_ori, model_forge_path, forging_path, device
                )
                xent_labels += xent_dic['labels']
                xent_original += xent_dic['original']
                xent_forged += xent_dic['forged']
                model_dists += model_dist
                exp_id += [delete_ind] * num_del  # [delete_ind] * num_del
            
            # xent_original = np.array(xent_original)
            # xent_forged = np.array(xent_forged)
            # model_dists = np.array(model_dists)
        
            torch.save({
                'exp_id': exp_id,
                'num_del': num_del,
                'xent_labels': xent_labels,
                'xent_original': xent_original,
                'xent_forged': xent_forged,
                'model_dists': model_dists
            }, output_file)
        
        else:
            thr = compute_or_load_threshold(
                None, None, None, device, 
                outfile='./exp-xent/{}/del-{}-epoch-{}/{}-thr.pkl'.format(dataset, num_del, n_epochs, forging_path)
            )
            print('loading pre-computed MI scores')
            dic = torch.load(output_file)
            xent_labels = dic['xent_labels']
            xent_original = dic['xent_original']
            xent_forged = dic['xent_forged']
            model_dists = dic['model_dists']
            num_del = dic['num_del']
            exp_id = dic['exp_id']
        
        df_dic["K"] += [splitK] * len(xent_original)
        df_dic["xent_labels"] += list(xent_labels)
        df_dic["xent_original"] += list(xent_original)
        df_dic["xent_forged"] += list(xent_forged)

        if eval_split == 'diff':
            df_dic["exp_id"] += exp_id
            df_dic[dist_name] += model_dists
            df_dic[xentdiff_name] += list(np.array(xent_original) - np.array(xent_forged))
            df_dic[xentdiv_name] += list(np.array(xent_forged) / np.array(xent_original) - 1.0)

        # print MI correctness statistics
        xent_original = np.array(xent_original) 
        xent_forged = np.array(xent_forged)
        
        # MI_res = {  # T is used, F is unused, 1st is ori, 2nd is forged
        #     'TT': sum((xent_original >= thr) * (xent_forged >= thr)),
        #     'TF': sum((xent_original >= thr) * (xent_forged <  thr)),
        #     'FT': sum((xent_original <  thr) * (xent_forged >= thr)),
        #     'FF': sum((xent_original <  thr) * (xent_forged <  thr)),
        # }

    df = pd.DataFrame(data=df_dic)
    df["K"] = df["K"].astype(CategoricalDtype(K_RANGE))

    # evaluate MI prediction (hard)
    for splitK in K_RANGE:
        print("K={}".format(splitK))
        df_K = df[df["K"] == splitK]

        # MI_pred_on_DIFF = {'exist_wrong': 0, 'total': 0}
        # for exp_id in range(0, int(Ntrain // num_del)):
        #     df_exp_id = df_K[df_K['exp_id'] == exp_id]
        #     if len(df_exp_id) == 0:
        #         continue
        #     if sum((df_exp_id["xent_original"] >= thr) ^ (df_exp_id["xent_forged"] >= thr)) > 0:
        #         MI_pred_on_DIFF['exist_wrong'] += 1
        #     MI_pred_on_DIFF['total'] += 1

        #     MI_pred_on_DIFF['exist_wrong'] += sum((df_exp_id["xent_original"] >= thr) ^ (df_exp_id["xent_forged"] >= thr))
        #     MI_pred_on_DIFF['total'] += len(df_exp_id["xent_original"])
        # print('{} out of {} models have >= 1 different MI pred on subset DIFF'.format(MI_pred_on_DIFF['exist_wrong'], MI_pred_on_DIFF['total']))

        wrong = 0
        for label in range(N_CLASSES):
            df_label = df_K[df_K['xent_labels'] == label]
            wrong += sum((df_label["xent_original"] >= thr[label]) ^ (df_label["xent_forged"] >= thr[label]))
        print('{} out of {} models have different MI pred'.format(
            wrong, len(df_K)
        ))

    # evaluate MI outputs (soft)
    outlier_scale = 3  # default = 1.5

    # plt.figure(figsize=(8,5))
    # df_unduplicate = df[df.index % num_del == 0]
    # q1, q3 = np.percentile(df_unduplicate[dist_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df_unduplicate[(df_unduplicate[dist_name] < high) & (df_unduplicate[dist_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=list(df_tmp[dist_name]), bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/model_dist-{}.png'.format(dataset, num_del, n_epochs, forging_path), dpi=300)

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[xentdiff_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[xentdiff_name] < high) & (df[xentdiff_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[xentdiff_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/xent_diff-{}.png'.format(dataset, num_del, n_epochs, forging_path), dpi=300)

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[xentdiv_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[xentdiv_name] < high) & (df[xentdiv_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[xentdiv_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/xent_div-{}.png'.format(dataset, num_del, n_epochs, forging_path), dpi=300)

    # quantiles = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
    # for col in [dist_name, xentdiff_name, xentdiv_name]:
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


if __name__ == "__main__":
    print('-'*30)

    # 1.7 Gb memory
    gpu = 1
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
    M = 400  # 200, 400
    num_del = 100  # 10, 100
    n_epochs = 20

    # eval_split = 'diff'

    recompute = False

    shadow_model = model

    print('-'*30)

    for eval_split in ['diff', 'train_diff', 'test']:
        print(model, eval_split)
        main(dataset, model, shadow_model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, device)
        print('-'*30)

    print('-'*30)
