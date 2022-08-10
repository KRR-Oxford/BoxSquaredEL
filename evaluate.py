from collections import Counter

import pandas as pd
import logging
import numpy as np
import torch
import math
from tqdm import trange, tqdm
from torch.nn.functional import softplus, relu

from RankingResult import RankingResult
from utils.utils import get_device
from utils.inferences_data_loader import load_data, load_inferences_data

logging.basicConfig(level=logging.INFO)


def main():
    evaluate('GALEN', 'inferences', embedding_size=200, ranking_fn='l2', beta=1, last=True)
    # evaluate('GO', embedding_size=200, ranking_fn='l1', beta=.5)
    # evaluate('ANATOMY', embedding_size=50, ranking_fn='l1', beta=.5)


def evaluate(dataset, task, embedding_size, beta, ranking_fn, last=False):
    which_model = 'last' if last else 'best'
    cls_embeds_file = f'data/{dataset}/{task}/class_embed_{which_model}.pkl'
    # rel_embeds_file = f'data/{dataset}/relationELEmbed.pkl'

    device = get_device()

    cls_df = pd.read_pickle(cls_embeds_file)
    # rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    # nb_relations = len(rel_df)
    # print(f'#Classes: {nb_classes}, #Relations: {nb_relations}')

    embeds_list = cls_df['embeddings'].values
    # r_embeds_list = rel_df['embeddings'].values
    # relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    assert (size == 2 * embedding_size)
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
    test_data = load_inferences_data(dataset, classes)

    acc = compute_accuracy(embeds, embedding_size, test_data, device)
    ranking = compute_ranks(embeds, embedding_size, test_data, device, ranking_fn, beta, use_tqdm=True)

    ranks_dict = Counter(ranking.ranks)
    rank_auc = compute_rank_roc(ranks_dict, nb_classes)

    print(f'{dataset}: acc: {acc:.3f}, top1: {ranking.top1:.2f}, top10: {ranking.top10:.2f}, '
          f'top100: {ranking.top100:.2f}, mean: {np.mean(ranking.ranks):.2f}, median: {np.median(ranking.ranks):.2f}, '
          f'auc: {rank_auc:.2f}')


def compute_accuracy(embeds, embedding_size, eval_data, device):
    offsets = torch.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]
    eval_data = torch.tensor(eval_data, requires_grad=False).long().to(device)

    c_embeds = embeds[eval_data[:, 0]]
    c_offsets = offsets[eval_data[:, 0]]
    d_embeds = embeds[eval_data[:, 1]]
    d_offsets = offsets[eval_data[:, 1]]

    euc = torch.abs(c_embeds - d_embeds)
    results = euc + c_offsets - d_offsets
    return (results <= 0).all(dim=1).float().mean()


def compute_accuracy2(embeds, embedding_size, eval_data, device):
    offsets = torch.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]

    acc = 0
    for (c, d) in eval_data:
        c_min = embeds[c] - offsets[c]
        c_max = embeds[c] + offsets[c]
        d_min = embeds[d] - offsets[d]
        d_max = embeds[d] + offsets[d]
        if (c_min >= d_min).all().item() and (c_max <= d_max).all().item():
            acc += 1

    print(f'Number correct: {acc}')
    return acc / len(eval_data)


def compute_ranks(embeds, embedding_size, eval_data, device, ranking_fn, beta, batch_size=100, use_tqdm=False):
    offsets = torch.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]

    top1 = 0.
    top10 = 0.
    top100 = 0.
    ranks = torch.Tensor().to(device)
    n = len(eval_data)

    num_batches = math.ceil(n / batch_size)
    eval_data = torch.tensor(eval_data, requires_grad=False).long().to(device)

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]
        batch_embeds = embeds[batch_data[:, 0]]

        if ranking_fn in ['l1', 'l2']:
            dists = batch_embeds[:, None, :] - torch.tile(embeds, (current_batch_size, 1, 1))
            order = 1 if ranking_fn == 'l1' else 2
            dists = torch.linalg.norm(dists, dim=2, ord=order)
        elif ranking_fn == 'softplus':
            batch_offsets = offsets[batch_data[:, 0]]
            eucs = torch.abs(batch_embeds[:, None, :] - torch.tile(embeds, (current_batch_size, 1, 1)))
            dists = eucs - offsets[None, :, :] + batch_offsets[:, None, :]
            dists = torch.linalg.norm(softplus(dists, beta=beta), dim=2)
            # dists = torch.linalg.norm(relu(dists), dim=2)
        else:
            raise ValueError('Illegal argument for ranking_fn')

        index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
        batch_ranks = torch.take_along_dim(index, batch_data[:, 1].reshape(-1, 1), dim=1).flatten()

        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks = torch.cat((ranks, batch_ranks))

    top1 /= n
    top10 /= n
    top100 /= n

    return RankingResult(top1, top10, top100, ranks.tolist())


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
