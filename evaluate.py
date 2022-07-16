from collections import Counter

import pandas as pd
import logging
import numpy as np
import torch
import math
from tqdm import trange

from utils.utils import get_device
from utils.el_data_loader import load_data, load_valid_data

logging.basicConfig(level=logging.INFO)


def main():
    dataset = 'GALEN'
    embedding_size = 50

    cls_embeds_file = f'data/{dataset}/classELEmbed_best.pkl'
    # rel_embeds_file = f'data/{dataset}/relationELEmbed.pkl'

    print('Evaluating')
    device = get_device()

    cls_df = pd.read_pickle(cls_embeds_file)
    # rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    # nb_relations = len(rel_df)
    # print(f'#Classes: {nb_classes}, #Relations: {nb_relations}')

    embeds_list = cls_df['embeddings'].values
    # classes = {v: k for k, v in enumerate(cls_df['classes'])}
    # r_embeds_list = rel_df['embeddings'].values
    # relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = torch.zeros((nb_classes, size), requires_grad=False).to(device)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = torch.from_numpy(emb).to(device)

    # relations
    # r_size = len(r_embeds_list[0])
    # r_embeds = torch.zeros((nb_relations, r_size), requires_grad=False).to(device)
    # for i, emb in enumerate(r_embeds_list):
    #     r_embeds[i, :] = torch.from_numpy(emb).to(device)

    print('Loading data')
    _, classes, relations = load_data(dataset)
    # train_file = f'data/{dataset}/{dataset}_train.txt'
    test_file = f'data/{dataset}/{dataset}_test.txt'
    test_data = load_valid_data(test_file, classes, relations)

    acc = compute_accuracy(embeds, embedding_size, test_data, device)
    top1, top10, top100, mean_rank, ranks = compute_ranks(embeds, embedding_size, test_data, device, use_tqdm=True)

    ranks_dict = Counter(ranks.tolist())
    rank_auc = compute_rank_roc(ranks_dict, nb_classes)

    print(f'{dataset}: acc: {acc:.2f}, top1: {top1:.2f}, top10: {top10:.2f}, '
          f'top100: {top100:.2f}, mean: {mean_rank:.2f}, median: {torch.median(ranks):.2f}, auc: {rank_auc:.2f}')


def compute_accuracy(embeds, embedding_size, eval_data, device):
    offsets = torch.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]
    eval_data = torch.tensor(eval_data, requires_grad=False).long().to(device)

    c_embeds = embeds[eval_data[:, 0]]
    c_offsets = offsets[eval_data[:, 0]]
    d_embeds = embeds[eval_data[:, 2]]
    d_offsets = offsets[eval_data[:, 2]]

    euc = torch.abs(c_embeds - d_embeds)
    results = euc + c_offsets - d_offsets
    results = torch.relu(results)
    results = results.sum(dim=1)
    return (results == 0).sum().item() / eval_data.shape[0]

def compute_ranks(embeds, embedding_size, eval_data, device, batch_size=100, use_tqdm=False):
    offsets = torch.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]

    top1 = 0.
    top10 = 0.
    top100 = 0.
    ranks = torch.Tensor().to(device)
    n = len(eval_data)

    num_batches = math.ceil(n / batch_size)
    eval_data = torch.tensor(eval_data, requires_grad=False).long().to(device)
    r = eval_data[0, 1]
    assert ((eval_data[:, 1] == r).sum() == n)  # assume we use the same r everywhere

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]

        batch_embeds = embeds[batch_data[:, 0]]
        dists = batch_embeds[:, None, :] - torch.tile(embeds, (current_batch_size, 1, 1))

        # batch_embeds = embeds[batch_data[:, 0]]
        # batch_offsets = offsets[batch_data[:, 0]]
        # eucs = torch.abs(batch_embeds[:, None, :] - torch.tile(embeds, (current_batch_size, 1, 1)))
        # dists = eucs - offsets[None, :, :] + batch_offsets[:, None, :]
        # dists = torch.relu(dists)

        # batch_starts = embeds[batch_data[:, 0]] - offsets[batch_data[:, 0]]
        # starts = embeds - offsets
        # starts = torch.maximum(batch_starts[:, None, :], torch.tile(starts, (current_batch_size, 1, 1)))
        # batch_ends = embeds[batch_data[:, 0]] + offsets[batch_data[:, 0]]
        # ends = embeds + offsets
        # ends = torch.minimum(batch_ends[:, None, :], torch.tile(ends, (current_batch_size, 1, 1)))
        # inter_vols =  torch.abs(starts - ends).prod(dim=)
        # log_inter_vols = torch.log(torch.relu(ends-starts) + 1e-3).sum(dim=2)
        # dists = -log_inter_vols

        dists = torch.linalg.norm(dists, dim=2, ord=1)  # TODO: ord=1
        index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
        batch_ranks = torch.take_along_dim(index, batch_data[:, 2].reshape(-1, 1), dim=1).flatten()

        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks = torch.cat((ranks, batch_ranks))

    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank = torch.mean(ranks)
    return top1, top10, top100, mean_rank, ranks


def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc


if __name__ == '__main__':
    main()
