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

from MI_forge_onesample_supportCIFAR import dataset_with_indices, flatten

# https://github.com/kuangliu/pytorch-cifar
import importlib 
from pytorch_cifar.models import (
    VGG, VGG_mini,
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet_mini,
    EfficientNetB0,
    MobileNet, MobileNetV2,
    ShuffleNet, ShuffleNetG2, ShuffleNetG3, ShuffleNetV2
)


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


def _m_entr_comp(probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1-probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)


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
        _outputs, _labels = _model_predictions(model_shadow, trainloader, device)
        tr_values = _m_entr_comp(_outputs, _labels)

        _outputs, _labels = _model_predictions(model_shadow, testloader, device)
        te_values = _m_entr_comp(_outputs, _labels)

        thr = _thre_setting(tr_values, te_values)
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

    mentr_dic = {'original': [], 'forged': []}  # i-th element in each value is i-th mentr; total len = num_del

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
    
    for sc_name, model in zip(['original', 'forged'], [model_ori, model_forge]):
        _outputs = model.forward(x_all) 
        _outputs = softmax_by_row(_outputs.data.cpu().numpy())
        mentr = _m_entr_comp(_outputs, _labels)
        mentr_dic[sc_name] = list(mentr)
    
    return mentr_dic, [checkpoint_forge['model_dist']] * num_del


def main(dataset, model, shadow_model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, 
            MOMENTUM, WEIGHT_DECAY, DO_ANNEAL, device):
    print('{}-del{}-epoch{}-M{}:\n'.format(dataset, num_del, n_epochs, M))
    print('momentum{}-wd{}-anneal{}'.format(MOMENTUM, WEIGHT_DECAY, DO_ANNEAL))
    print('eval_split:', eval_split)

    dist_name = r"$\|\theta_*-\theta_{-i}\|_2^2$"
    mentrdiff_name = r"$\Lambda_i(\theta_*)-\Lambda_i(\theta_{-i})$"
    mentrdiv_name = r"$\frac{\Lambda_i(\theta_{-i})}{\Lambda_i(\theta_*)}-1$"
    df_dic = {
        "K": [],
        "mentr_original": [],
        "mentr_forged": []
    }
    if eval_split == 'diff':
        df_dic = {
            **df_dic, 
            "exp_id": [],
            dist_name: [],
            mentrdiff_name: [],
            mentrdiv_name: []
        }

    for splitK in K_RANGE:
        print("K={}".format(splitK))

        # get output path ready
        forging_path = "{}-{}-Ntrain{}-momentum{}-weightdecay{}-anneal{}-K{}-M{}".format(
            dataset.split('-')[0], model, Ntrain, MOMENTUM, WEIGHT_DECAY, DO_ANNEAL, splitK, M)
        output_file = './exp/{}/del-{}-epoch-{}/{}-split-{}.pkl'.format(dataset, num_del, n_epochs, forging_path, eval_split)
        os.makedirs('./exp/{}/del-{}-epoch-{}'.format(dataset, num_del, n_epochs), exist_ok=True)

        if recompute:
            if dataset.startswith('cifar'):
                N_CLASSES = 100 if dataset.startswith('cifar100') else 10
                
                def _get_model(MODEL):
                    if MODEL == 'LeNet5':
                        model_obj = LeNet5(N_CLASSES, in_channels=3).to(device)
                    elif MODEL == 'LR':
                        raise NotImplementedError
                    elif MODEL.startswith('VGG') and not MODEL.endswith('mini'):
                        model_obj = VGG(MODEL).to(device)
                    elif MODEL.startswith('ShuffleNetV2'):
                        model_obj = ShuffleNetV2(float(MODEL.split('-')[1])).to(device)
                    else:
                        module = importlib.import_module('pytorch_cifar.models')
                        class_ = getattr(module, MODEL)
                        model_obj = class_().to(device)
                    return model_obj

                model_ori = _get_model(model)
                model_shadow = _get_model(shadow_model)

                data_folder = os.path.join('/tmp2', dataset.split('-')[0])
                transform_fn = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

                # add train/test loader to compute thr via shadow model
                trainloader = DataLoader(
                    dataset=datasets.CIFAR10(root=data_folder, train=True, transform=transform_fn, download=True),
                    batch_size=64, shuffle=False, drop_last=False)
                testloader = DataLoader(
                    dataset=datasets.CIFAR10(root=data_folder, train=False, transform=transform_fn, download=True),
                    batch_size=64, shuffle=False, drop_last=False)

                CIFARWithIndices = dataset_with_indices(datasets.CIFAR10)
                full_dataset = CIFARWithIndices(root=data_folder, 
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
                "model_ori_state_dict-{}.pkl".format(n_epochs)
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
                outfile='./exp/{}/del-{}-epoch-{}/{}-thr.pkl'.format(dataset, num_del, n_epochs, forging_path)
            )

            mentr_original = []
            mentr_forged = []
            model_dists = []
            exp_id = []

            for delete_ind in tqdm(range(0, int(Ntrain // num_del))):  # int(Ntrain // num_del)
                model_forge_path = os.path.join(
                    "/tmp3/USERNAME/MI_project/Forgeability/tensorboard_delete_{}/".format(num_del),
                    forging_path,
                    "recovered-{}/del-{}".format(n_epochs, delete_ind),
                    "recovered.pkl"
                )

                if not os.path.exists(model_forge_path):
                    continue

                # print('delete {}:'.format(delete))
                mentr_dic, model_dist = compute_MI_score(
                    dataset, dataloader, Ntrain, delete_ind, num_del, eval_split,
                    model, model_ori, model_forge_path, forging_path, device
                )
                mentr_original += mentr_dic['original']
                mentr_forged += mentr_dic['forged']
                model_dists += model_dist
                exp_id += [delete_ind] * num_del  # [delete_ind] * num_del
            
            # mentr_original = np.array(mentr_original)
            # mentr_forged = np.array(mentr_forged)
            # model_dists = np.array(model_dists)
        
            torch.save({
                'exp_id': exp_id,
                'num_del': num_del,
                'mentr_original': mentr_original,
                'mentr_forged': mentr_forged,
                'model_dists': model_dists
            }, output_file)
        
        else:
            thr = compute_or_load_threshold(
                None, None, None, device, 
                outfile='./exp/{}/del-{}-epoch-{}/{}-thr.pkl'.format(dataset, num_del, n_epochs, forging_path)
            )
            print('loading pre-computed MI scores')
            dic = torch.load(output_file)
            mentr_original = dic['mentr_original']
            mentr_forged = dic['mentr_forged']
            model_dists = dic['model_dists']
            num_del = dic['num_del']
            exp_id = dic['exp_id']
        
        df_dic["K"] += [splitK] * len(mentr_original)
        df_dic["mentr_original"] += list(mentr_original)
        df_dic["mentr_forged"] += list(mentr_forged)

        if eval_split == 'diff':
            df_dic["exp_id"] += exp_id
            df_dic[dist_name] += model_dists
            df_dic[mentrdiff_name] += list(np.array(mentr_original) - np.array(mentr_forged))
            df_dic[mentrdiv_name] += list(np.array(mentr_forged) / np.array(mentr_original) - 1.0)

        # print MI correctness statistics
        mentr_original = np.array(mentr_original) 
        mentr_forged = np.array(mentr_forged)
        
        # MI_res = {  # T is used, F is unused, 1st is ori, 2nd is forged
        #     'TT': sum((mentr_original >= thr) * (mentr_forged >= thr)),
        #     'TF': sum((mentr_original >= thr) * (mentr_forged <  thr)),
        #     'FT': sum((mentr_original <  thr) * (mentr_forged >= thr)),
        #     'FF': sum((mentr_original <  thr) * (mentr_forged <  thr)),
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
        #     if sum((df_exp_id["mentr_original"] >= thr) ^ (df_exp_id["mentr_forged"] >= thr)) > 0:
        #         MI_pred_on_DIFF['exist_wrong'] += 1
        #     MI_pred_on_DIFF['total'] += 1

        #     MI_pred_on_DIFF['exist_wrong'] += sum((df_exp_id["mentr_original"] >= thr) ^ (df_exp_id["mentr_forged"] >= thr))
        #     MI_pred_on_DIFF['total'] += len(df_exp_id["mentr_original"])
        # print('{} out of {} models have >= 1 different MI pred on subset DIFF'.format(MI_pred_on_DIFF['exist_wrong'], MI_pred_on_DIFF['total']))

        print('{} out of {} models have different MI pred'.format(
            sum((df_K["mentr_original"] >= thr) ^ (df_K["mentr_forged"] >= thr)),
            len(df_K)
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
    # q1, q3 = np.percentile(df[mentrdiff_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[mentrdiff_name] < high) & (df[mentrdiff_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[mentrdiff_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/mentr_diff-{}.png'.format(dataset, num_del, n_epochs, forging_path), dpi=300)

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[mentrdiv_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[mentrdiv_name] < high) & (df[mentrdiv_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[mentrdiv_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/mentr_div-{}.png'.format(dataset, num_del, n_epochs, forging_path), dpi=300)

    # quantiles = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
    # for col in [dist_name, mentrdiff_name, mentrdiv_name]:
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

    # 1.2 Gb memory
    gpu = 1
    device = torch.device('cuda:{}'.format(gpu))

    # dataset = 'cifar10-10k'
    # # model = 'ResNet_mini'  # "ResNet_mini" "VGG_mini" "ShuffleNetV2-0.5"
    # K_RANGE = ['2']  # compare between hyperparams
    # Ntrain = 10000
    # M = 200
    # num_del = 1
    # n_epochs = 20

    dataset = 'cifar10-50k'
    model = 'ResNet_mini'  # "ResNet_mini" "VGG_mini" "ShuffleNetV2-0.5"
    K_RANGE = ['2']  # compare between hyperparams
    Ntrain = 50000
    M = 200
    num_del = 10
    n_epochs = 20

    # MOMENTUM = 0.0  # 0.9
    # WEIGHT_DECAY = 0.0  # 0.0005
    # DO_ANNEAL = False  # True

    MOMENTUM = 0.9  # 0.9
    WEIGHT_DECAY = 0.0005  # 0.0005
    DO_ANNEAL = True  # True

    # eval_split = 'diff'

    recompute = False

    for model in ["VGG_mini", "ResNet_mini"]:
        if model == "VGG_mini":
            shadow_model = "ResNet_mini"
        elif model == "ResNet_mini":
            shadow_model = "VGG_mini"

        for eval_split in ['diff', 'train_diff', 'test']:
            print(model, eval_split)
            main(dataset, model, shadow_model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, 
                MOMENTUM, WEIGHT_DECAY, DO_ANNEAL, device)
            print('-'*30)

    print('-'*30)
