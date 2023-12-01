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


def main_old(args):
    indices = torch.load(args.ind_file)
    ind_forge = indices['forge']
    
    # Two-sample Kolmogorov–Smirnov test
    assert len(ind_forge) % args.epochs == 0
    n_iter_per_epoch = len(ind_forge) // args.epochs

    freq_SGD = np.array(list(range(args.Ndel, args.N)) * args.epochs)
    freq_forge = []
    T_ks_stat, T_ks_pvalue = [[], []], [[], []]
    for i, freq_at_iter in tqdm(enumerate(ind_forge)):
        freq_forge += freq_at_iter
        if (i + 1) % n_iter_per_epoch == 0:
            # forged
            T_ks = stats.kstest(freq_forge, freq_SGD)
            T_ks_stat[0].append(T_ks.statistic)
            T_ks_pvalue[0].append(T_ks.pvalue)

            # SGD
            freq_SGD = np.random.choice(args.N - args.Ndel, len(freq_forge), replace=True) + args.Ndel
            T_ks = stats.kstest(freq_SGD, freq_SGD)
            T_ks_stat[1].append(T_ks.statistic)
            T_ks_pvalue[1].append(T_ks.pvalue)
    
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs = axs.flatten()
    axs[0].plot(np.arange(len(T_ks_stat[0])), T_ks_stat[0], 
        color='blue', linestyle='dashed', label='KS statistics (forged)')
    axs[0].plot(np.arange(len(T_ks_stat[0])), T_ks_pvalue[0], 
        color='orange', linestyle='dashed', label='KS p value (forged)')
    axs[0].plot(np.arange(len(T_ks_stat[1])), T_ks_stat[1], 
        color='blue', linestyle='solid', label='KS statistics (SGD)')
    axs[0].plot(np.arange(len(T_ks_stat[1])), T_ks_pvalue[1], 
        color='orange', linestyle='solid', label='KS p value (SGD)')
    axs[0].set_xlabel('num epochs')
    axs[0].set_title('Kolmogorov–Smirnov test of forged batches')
    axs[0].legend()

    freq_SGD = np.random.choice(args.N - args.Ndel, len(freq_forge), replace=True) + args.Ndel
    freq_forge
    distr_forge, distr_SGD = np.zeros(args.N - args.Ndel), np.zeros(args.N - args.Ndel)
    for i in freq_forge:
        distr_forge[i-args.Ndel] += 1
        distr_SGD[np.random.randint(args.N - args.Ndel)] += 1
    # distr_forge /= sum(distr_forge)
    distr_forge.sort()
    # distr_SGD /= sum(distr_SGD)
    distr_SGD.sort()
    axs[1].plot(np.arange(len(distr_forge)), distr_forge, 
        color='blue', linestyle='solid', label='sorted frequency (forged)')
    axs[1].plot(np.arange(len(distr_SGD)), distr_SGD, 
        color='orange', linestyle='solid', label='sorted frequency (SGD)')
    axs[1].set_ylabel('frequency')
    axs[1].set_title('Sorted frequency of forged batches')
    axs[1].legend()
    plt.savefig(args.ind_file + '-KS.png', dpi=400)


def main(args):

    os.makedirs('batchtest', exist_ok=True)
    output_file = os.path.join('batchtest', '-'.join(args.ind_file.split('/')))

    UNIFORM = np.ones(args.N - args.Ndel) / (args.N - args.Ndel)

    if not os.path.exists(output_file):
        print('loading forged inds from:', args.ind_file)
        dic = torch.load(args.ind_file, map_location='cpu')
        forge_inds = dic["forge_inds"]
        
        # Instead of Two-sample Kolmogorov–Smirnov test, do L1 distance
        print('running batch frequency test')
        l1_dist = []
        for exp_id, row in tqdm(enumerate(forge_inds[:args.first_n_exp])):
            # compute frequencies
            # deleted = list(range(exp_id * args.Ndel, (exp_id + 1) * args.Ndel))

            freq_forge = np.array([0.0 for _ in range(args.N - args.Ndel)])
            for minibatch in row[:int(args.N // args.batchsize * args.epochs)]:
                for ind in minibatch:
                    if ind >= (exp_id + 1) * args.Ndel:
                        freq_forge[ind - args.Ndel] += 1.0
                    elif ind < exp_id * args.Ndel:
                        freq_forge[ind] += 1
                    else:
                        raise ValueError
            freq_forge /= sum(freq_forge)
            freq_forge.sort()

            l1_dist.append(np.linalg.norm(freq_forge - UNIFORM, ord=1))

            # T_ks = stats.kstest(freq_forge, UNIFORM)
            # T_ks_stat.append(T_ks.statistic)
            # T_ks_pvalue.append(T_ks.pvalue)
        
        output_dic = {"l1_dist": l1_dist}
        torch.save(output_dic, output_file)
    
    else:
        print('loading pre-computed results from:', output_file)
        output_dic = torch.load(output_file)
        l1_dist = output_dic["l1_dist"]

        # T_ks_stat, T_ks_pvalue = T_ks["T_ks_stat"], T_ks["T_ks_pvalue"]
    
    # generate freq_SGD by simulating SGD
    print("simulating SGD")
    l1_dist_simulatedSGD = []
    for _ in range(10):
        freq_SGD = np.array([0.0 for _ in range(args.N - args.Ndel)])
        for _ in range(int(args.N / args.batchsize) * args.epochs):
            minibatch = np.random.choice(args.N - args.Ndel, size=args.batchsize, replace=False)
            for ind in minibatch:
                freq_SGD[ind] += 1.0
        freq_SGD /= sum(freq_SGD)
        freq_SGD.sort()
        l1_dist_simulatedSGD.append(np.linalg.norm(freq_SGD - UNIFORM, ord=1))

    print("plotting batch frequency l1 dist from uniform")
    df_dic = {
        "algorithm": ["forged"] * len(l1_dist) + ["SGD"] * len(l1_dist_simulatedSGD),
        "L1 distance to uniform": l1_dist + l1_dist_simulatedSGD
    }
    df = pd.DataFrame(data=df_dic)
    plt.figure(figsize=(8,5))
    sns.violinplot(y=df["algorithm"], x=df["L1 distance to uniform"])
    plt.savefig('{}-L1dist_from_uniform.png'.format(output_file), dpi=300)

    # print('scatter plot KS_stat vs KS_pvalue')
    # plt.figure(figsize=(8,6))
    # # sns.scatterplot(data=np.vstack([T_ks_stat, T_ks_pvalue]).T)
    # plt.scatter(x=T_ks_stat, y=T_ks_pvalue, s=1, c='blue', alpha=0.2)
    # plt.xlabel('KS statistics')
    # plt.ylabel('p-values')
    # plt.savefig('{}-KS.png'.format(output_file), dpi=600)

    # print('frequency comparison')
    # plt.figure(figsize=(8,6))
    # xs = list(range(len(all_freq_forge[0])))
    # s, alpha = 0.5, 0.4
    # for i, freq_forge in enumerate(all_freq_forge):
    #     plt.scatter(xs, freq_forge, s=s, alpha=alpha, label="del {}th subset".format(i))
    # plt.scatter(xs, freq_SGD, s=s, alpha=alpha, label="Simulated SGD")
    # plt.scatter(xs, np.ones(len(freq_SGD)) / len(freq_SGD), s=s, alpha=alpha, label="UNIFORM")
    # plt.legend(ncol=2)
    # plt.xlabel("i-th smallest frequency")
    # plt.ylabel("frequency")
    # plt.savefig('{}-freq.png'.format(output_file), dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_file", type=str, help="pkl file to store indices")
    parser.add_argument("--N", type=int, default=5000, help="number of samples to train")
    parser.add_argument("--Ndel", type=int, default=100, help="number of samples to delete")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batchsize", type=int, default=100, help="batchsize")
    parser.add_argument("--first_n_exp", type=int, default=10, help="first n experiments")


    args = parser.parse_args()
    assert 0 < args.Ndel < args.N 
    assert os.path.exists(args.ind_file)

    main(args)