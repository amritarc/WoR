import numpy as np
from datetime import datetime 
import os
from tqdm import tqdm
from copy import deepcopy
import scipy
# from scipy.stats import entropy
import argparse
import sys
# sys.path.append("../../Forgeability")

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

from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModel


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


@torch.no_grad()
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
def get_MI_thr(dataset, Ntrain, train_ds, test_ds, population_ds, model_ori, FPR, device):
    """
    Perform MI-attackP for model_ori on dataset and return threshold
    """

    fpr_tolerance_list = [FPR]

    # load dataset
    if dataset.startswith('mnist'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # create the target model's dataset
    target_dataset = Dataset(
        data_dict={'train': train_ds, 'test': test_ds},
        default_input='x', default_output='y'
    )

    # create the reference dataset
    reference_dataset = Dataset(
        data_dict={'train': population_ds},
        default_input='x', default_output='y'
    )

    # compute scores
    target_model = PytorchModel(model_obj=model_ori, loss_fn=loss_fn)

    target_info_source = InformationSource(
        models=[target_model], 
        datasets=[target_dataset]
    )
    reference_info_source = InformationSource(
        models=[target_model],
        datasets=[reference_dataset]
    )
    audit_obj = Audit(
        metrics=MetricEnum.POPULATION,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=fpr_tolerance_list
    )
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]
    result = audit_results[0]
    print('threshold:', result.threshold)

    from privacy_meter import audit_report
    audit_report.REPORT_FILES_DIR = './privacy_meter/report_files'
    SignalHistogramReport.generate_report(
        metric_result=result,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        show=True,
        save=True
    )
    
    return result.threshold


@torch.no_grad()
def compute_MI_score(
    dataset, Ntrain, eval_split,
    train_ds, test_ds, population_ds,
    delete_ind, num_del, 
    model_ori, model_forge, model_forge_path, forging_path,
    FPR, 
    device
):
    """
    Perform ImprovedMI-attackP for model on eval_split
        diff: from delete_ind * num_del to (delete_ind+1) * num_del
        train_diff: random points in train but not in diff
        test: random points in test
    """

    assert eval_split in ['diff', 'train_diff', 'test']
    fpr_tolerance_list = [FPR]

    # load dataset
    if dataset.startswith('mnist'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # create the target model's dataset (as placeholder; meaningless)
    target_dataset = Dataset(
        data_dict={'train': train_ds, 'test': test_ds},
        default_input='x', default_output='y'
    )

    # create the reference dataset
    reference_dataset = Dataset(
        data_dict={'train': population_ds},
        default_input='x', default_output='y'
    )

    # load model
    checkpoint_forge = torch.load(model_forge_path, map_location='cpu')
    # model_forge = deepcopy(model_ori)
    model_forge.load_state_dict(checkpoint_forge['model_state_dict'])
    model_forge.eval()

    score_dic = {'original': [], 'forged': []}  # i-th element in each value is i-th score; total len = num_del

    for sc_name, model in zip(['original', 'forged'], [model_ori, model_forge]):
        target_model = PytorchModel(model_obj=model, loss_fn=loss_fn)

        target_info_source = InformationSource(
            models=[target_model], 
            datasets=[target_dataset]
        )
        reference_info_source = InformationSource(
            models=[target_model],
            datasets=[reference_dataset]
        )
        audit_obj = Audit(
            metrics=MetricEnum.POPULATION,
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            target_info_sources=target_info_source,
            reference_info_sources=reference_info_source,
            fpr_tolerances=fpr_tolerance_list
        )
        audit_obj.prepare()
        audit_results = audit_obj.run()[0]
        result = audit_results[0]
        if eval_split == 'diff':
            assert len(result.reference_signal_values) == num_del
        score_dic[sc_name] += list(result.reference_signal_values)      
    
    return score_dic, [checkpoint_forge['model_dist']] * num_del


def main(dataset, model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, device):

    FPR = 0.1

    print('{}-del{}-epoch{}-M{}:\n'.format(dataset, num_del, n_epochs, M))
    print('eval_split:', eval_split)

    # initialize dict
    dist_name = r"$\|\theta_*-\theta_{-i}\|_2^2$"
    scorediff_name = r"$\Lambda_i(\theta_*)-\Lambda_i(\theta_{-i})$"
    scorediv_name = r"$\frac{\Lambda_i(\theta_{-i})}{\Lambda_i(\theta_*)}-1$"
    df_dic = {
        "K": [],
        "score_original": [],
        "score_forged": []
    }
    if eval_split == 'diff':
        df_dic = {
            **df_dic, 
            "exp_id": [],
            dist_name: [],
            scorediff_name: [],
            scorediv_name: []
        }

    for splitK in K_RANGE:
        print("K={}".format(splitK))

        # get output path ready
        forging_path = "{}-{}-Ntrain{}-K{}-M{}".format(dataset.split('-')[0], model, Ntrain, splitK, M)
        output_file = './exp/{}/del-{}-epoch-{}/{}-split-{}.pkl'.format(dataset, num_del, n_epochs, forging_path, eval_split)
        os.makedirs('./exp/{}/del-{}-epoch-{}'.format(dataset, num_del, n_epochs), exist_ok=True)

        if recompute:
            N_CLASSES = 10
            model_ori = LeNet5(N_CLASSES).to(device)
            model_forge = deepcopy(model_ori)

            model_ori_path = os.path.join(
                "../../Forgeability/tensorboard_delete_{}/".format(num_del),
                forging_path,
                "recovered-{}".format(n_epochs),
                "model_ori_state_dict.pkl"  # "forge.pkl"
            )
            checkpoint_ori = torch.load(model_ori_path, map_location='cpu')
            model_ori.load_state_dict(checkpoint_ori['model_ori_state_dict'])
            model_ori.eval()

            # load dataset 
            if dataset.startswith('mnist'):
                if model == 'LeNet5':
                    IMG_SIZE = 32
                else:
                    raise NotImplementedError
                transform_fn = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
                full_dataset = datasets.MNIST(
                    root=os.path.join('/tmp2', dataset.split('-')[0]), 
                    train=True, 
                    transform=transform_fn,
                    download=True
                )
                train_dataset = Subset(full_dataset, list(range(Ntrain)))
                trainloader_entire = DataLoader(
                    dataset=train_dataset, 
                    batch_size=len(train_dataset), 
                    shuffle=False
                )

                test_dataset = datasets.MNIST(
                    root=os.path.join('/tmp2', dataset.split('-')[0]), 
                    train=False, 
                    transform=transform_fn,
                    download=True
                )
                testloader_entire = DataLoader(
                    dataset=train_dataset, 
                    batch_size=len(train_dataset), 
                    shuffle=False
                )
            else:
                raise NotImplementedError
            
            # extract population (deleted samples)
            x_train_all, y_train_all = next(iter(trainloader_entire))
            x_train_all, y_train_all = x_train_all.to(device), y_train_all.to(device)
            x_test_all, y_test_all = next(iter(testloader_entire))
            x_test_all, y_test_all = x_test_all.to(device), y_test_all.to(device)
            
            # create the target and reference models' dataset
            train_ds = {'x': x_train_all[:1000], 'y': y_train_all[:1000]}
            test_ds = {'x': x_test_all[:1000], 'y': y_test_all[:1000]}
            population_ds = {'x': x_train_all[:10], 'y': y_train_all[:10]}  # (as placeholder; meaningless)

            # compute threshold
            with torch.no_grad():
                thr = get_MI_thr(dataset, Ntrain, train_ds, test_ds, population_ds, model_ori, FPR, device)
            del train_ds, test_ds, population_ds

            score_original = []
            score_forged = []
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

                # create the target and reference models' dataset
                train_ds = {'x': x_train_all[:10], 'y': y_train_all[:10]}  # (as placeholder; meaningless)
                test_ds = {'x': x_train_all[:10], 'y': y_train_all[:10]}  # (as placeholder; meaningless)
                if eval_split == 'diff':
                    x_population = x_train_all[delete_ind * num_del:(delete_ind+1) * num_del]
                    y_population = y_train_all[delete_ind * num_del:(delete_ind+1) * num_del]
                elif eval_split == 'train_diff':
                    _train_diff_indices = list(set(list(range(Ntrain))) - set(list(range(delete_ind * num_del, (delete_ind+1) * num_del))))
                    indices_to_compute_MI = np.random.choice(_train_diff_indices, 5*num_del, replace=False)
                    x_population = x_train_all[indices_to_compute_MI]
                    y_population = y_train_all[indices_to_compute_MI]
                elif eval_split == 'test':
                    indices_to_compute_MI = np.random.choice(list(range(x_test_all.shape[0])), 5*num_del, replace=False)
                    x_population = x_test_all[indices_to_compute_MI]
                    y_population = y_test_all[indices_to_compute_MI]

                population_ds = {'x': x_population, 'y': y_population}

                score_dic, model_dist = compute_MI_score(
                    dataset, Ntrain, eval_split,
                    train_ds, test_ds, population_ds,
                    delete_ind, num_del, 
                    model_ori, model_forge, model_forge_path, forging_path, 
                    FPR,
                    device
                )
                score_original += score_dic['original']
                score_forged += score_dic['forged']
                model_dists += model_dist
                exp_id += [delete_ind] * num_del
            
            # score_original = np.array(score_original)
            # score_forged = np.array(score_forged)
            # model_dists = np.array(model_dists)
        
            torch.save({
                'exp_id': exp_id,
                'num_del': num_del,
                'score_original': score_original,
                'score_forged': score_forged,
                'model_dists': model_dists,
                'thr': thr
            }, output_file)
        
        else:
            dic = torch.load(output_file)
            score_original = dic['score_original']
            score_forged = dic['score_forged']
            model_dists = dic['model_dists']
            num_del = dic['num_del']
            exp_id = dic['exp_id']
            thr = dic['thr']
        
        df_dic["K"] += [splitK] * len(score_original)
        df_dic["score_original"] += list(score_original)
        df_dic["score_forged"] += list(score_forged)

        if eval_split == 'diff':
            df_dic["exp_id"] += exp_id
            df_dic[dist_name] += model_dists
            df_dic[scorediff_name] += list(np.array(score_original) - np.array(score_forged))
            df_dic[scorediv_name] += list(np.array(score_forged) / np.array(score_original) - 1.0)

        score_original = np.array(score_original) 
        score_forged = np.array(score_forged)


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
        #     if sum((df_exp_id["score_original"] >= thr) ^ (df_exp_id["score_forged"] >= thr)) > 0:
        #         MI_pred_on_DIFF['exist_wrong'] += 1
        #     MI_pred_on_DIFF['total'] += 1

        #     MI_pred_on_DIFF['exist_wrong'] += sum((df_exp_id["score_original"] >= thr) ^ (df_exp_id["score_forged"] >= thr))
        #     MI_pred_on_DIFF['total'] += len(df_exp_id["score_original"])
        # print('{} out of {} models have >= 1 different MI pred on subset DIFF'.format(MI_pred_on_DIFF['exist_wrong'], MI_pred_on_DIFF['total']))

        print('{} out of {} models have different MI pred'.format(
            sum((df_K["score_original"] >= thr) ^ (df_K["score_forged"] >= thr)),
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
    # q1, q3 = np.percentile(df[scorediff_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[scorediff_name] < high) & (df[scorediff_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[scorediff_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/score_diff-M{}.png'.format(dataset, num_del, n_epochs, M), dpi=300)

    # plt.figure(figsize=(8,5))
    # q1, q3 = np.percentile(df[scorediv_name], [25, 75])
    # low, high = q1 - (q3 - q1) * outlier_scale, q3 + (q3 - q1) * outlier_scale
    # df_tmp = df[(df[scorediv_name] < high) & (df[scorediv_name] > low)]
    # sns.violinplot(y=df_tmp["K"], x=df_tmp[scorediv_name], bw=0.1)
    # plt.savefig('./exp/{}/del-{}-epoch-{}/score_div-M{}.png'.format(dataset, num_del, n_epochs, M), dpi=300)

    # quantiles = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
    # for col in [dist_name, scorediff_name, scorediv_name]:
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

    gpu = 1
    device = torch.device('cuda:{}'.format(gpu))

    # 3.1 GB memory

    recompute = False 
    # eval_split = 'train_diff'

    # dataset = 'mnist-10k'
    # model = 'LeNet5'
    # K_RANGE = ['5', '10']  # compare between hyperparams
    # Ntrain = 10000
    # M = 200
    # num_del = 1
    # n_epochs = 200

    # for eval_split in ['diff', 'train_diff', 'test']:
    #     print(model, eval_split)

    #     main(dataset, model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, device)

    #     print('-'*30)

    dataset = 'mnist-60k'
    model = 'LeNet5'
    K_RANGE = ['5', '10']  # compare between hyperparams
    Ntrain = 60000
    # M = 200  # 200, 400
    # num_del = 10  # 10, 100
    n_epochs = 20

    for num_del in [10, 100]:
        for M in [200, 400]:
            if num_del == 10 and M == 400:
                continue
            for eval_split in ['diff', 'train_diff', 'test']:
                print(model, eval_split)

                main(dataset, model, K_RANGE, Ntrain, M, num_del, n_epochs, eval_split, recompute, device)

                print('-'*30)

    print('-'*30)

