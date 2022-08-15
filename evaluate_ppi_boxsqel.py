from collections import Counter

import json
import logging
from tqdm import trange
import numpy as np
import torch
from loaded_models import LoadedModel
from torch.nn.functional import softplus, relu
from utils.ppi_data_loader import load_protein_data

import math
from utils.utils import get_device, memory

logging.basicConfig(level=logging.INFO)


def main():
    print('Evaluating')
    embedding_size = 200
    dataset = 'human'

    device = get_device()
    model = LoadedModel.from_name('boxsqel', f'data/ppi/{dataset}/boxsqel/', embedding_size, device, best=True)
    with open(f'data/ppi/{dataset}/classes.json', 'r') as f:
        classes = json.load(f)
    with open(f'data/ppi/{dataset}/relations.json', 'r') as f:
        relations = json.load(f)

    print('Loading data')
    prot_index, prot_dict = load_protein_index(classes)
    train_labels = load_train_labels(dataset, classes, relations, prot_dict, device)
    test_data = load_protein_data(dataset, 'test', classes, relations)

    ranks, top1, top10, top100, franks, ftop1, ftop10, ftop100 = \
        compute_ranks(model, test_data, prot_index, prot_dict, device, 'l2', train_labels, use_tqdm=True)

    ranks_dict = Counter(ranks.tolist())
    franks_dict = Counter(franks.tolist())
    rank_auc = compute_rank_roc(ranks_dict, len(prot_dict))
    frank_auc = compute_rank_roc(franks_dict, len(prot_dict))

    ranks = ranks.cpu().numpy()
    franks = franks.cpu().numpy()

    print(f'{dataset} {embedding_size} {top10:.2f} {top100:.2f} {np.mean(ranks):.2f} {np.median(ranks)} {rank_auc:.2f}')
    print(f'{dataset} {embedding_size} {ftop10:.2f} {ftop100:.2f} {np.mean(franks):.2f} {np.median(franks)} {frank_auc:.2f}')


def compute_ranks(model, eval_data, prot_index, prot_dict, device, ranking_fn, train_labels=None, use_tqdm=False):
    top1 = 0.
    top10 = 0.
    top100 = 0.
    ftop1 = 0.
    ftop10 = 0.
    ftop100 = 0.
    ranks = torch.Tensor().to(device)
    franks = torch.Tensor().to(device)
    n = len(eval_data)

    batch_size = 100
    num_batches = math.ceil(n / batch_size)
    eval_data = [(prot_dict[c], r, prot_dict[d]) for c, r, d in eval_data]
    eval_data = torch.tensor(eval_data, requires_grad=False).to(device)
    r = eval_data[0, 1]
    assert ((eval_data[:, 1] == r).sum() == n)  # assume we use the same r everywhere

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]

        class_boxes = model.get_boxes(model.class_embeds)
        bumps = model.bumps
        head_boxes = model.get_boxes(model.relation_heads)
        tail_boxes = model.get_boxes(model.relation_tails)

        centers = class_boxes.centers
        prot_centers = centers[prot_index]
        prot_bumps = bumps[prot_index]
        d_centers = prot_centers[batch_data[:, 2]]
        d_bumps = prot_bumps[batch_data[:, 2]]
        batch_heads = head_boxes.centers[batch_data[:, 1]]
        batch_tails = tail_boxes.centers[batch_data[:, 1]]

        bumped_c_centers = torch.tile(prot_centers, (current_batch_size, 1, 1)) + d_bumps[:, None, :]
        bumped_d_centers = d_centers[:, None, :] + torch.tile(prot_bumps, (current_batch_size, 1, 1))

        c_dists = bumped_c_centers - batch_heads[:, None, :]
        c_dists = torch.linalg.norm(c_dists, dim=2, ord=2)
        d_dists = bumped_d_centers - batch_tails[:, None, :]
        d_dists = torch.linalg.norm(d_dists, dim=2, ord=2)
        dists = c_dists + d_dists

        index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
        batch_ranks = torch.take_along_dim(index, batch_data[:, 0].reshape(-1, 1), dim=1).flatten()

        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks = torch.cat((ranks, batch_ranks))

        if train_labels is not None:
            dists = dists * train_labels[r.item()][batch_data[:, 2]]
            index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
            batch_ranks = torch.take_along_dim(index, batch_data[:, 0].reshape(-1, 1), dim=1).flatten()

            ftop1 += (batch_ranks <= 1).sum()
            ftop10 += (batch_ranks <= 10).sum()
            ftop100 += (batch_ranks <= 100).sum()
            franks = torch.cat((franks, batch_ranks))

    top1 /= n
    top10 /= n
    top100 /= n
    ftop1 /= n
    ftop10 /= n
    ftop100 /= n

    return ranks, top1, top10, top100, franks, ftop1, ftop10, ftop100


@memory.cache
def load_train_labels(dataset, classes, relations, prot_dict, device):
    train_data = load_protein_data(dataset, 'train', classes, relations)
    # valid_data = load_protein_data(dataset, 'valid', classes, relations)
    train_labels = {}
    for c, r, d in train_data:
        c, r, d = prot_dict[c], r, prot_dict[d]
        if r not in train_labels:
            train_labels[r] = torch.ones((len(prot_dict), len(prot_dict)), requires_grad=False).to(device)
        train_labels[r][c, d] = torch.inf
    return train_labels


@memory.cache
def load_protein_index(classes):
    proteins = {}
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            proteins[k] = v

    prot_index = list(proteins.values())
    prot_dict = {v: k for k, v in enumerate(prot_index)}
    return prot_index, prot_dict


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
