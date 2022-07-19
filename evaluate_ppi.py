from collections import Counter

import click as ck
import pandas as pd
import logging
from tqdm import trange
import numpy as np
import torch
from torch.nn.functional import softplus

from sklearn.metrics import roc_curve, auc
import math
from utils.utils import get_device, memory

logging.basicConfig(level=logging.INFO)

dataset = 'yeast'
dataset_id = 4932 if dataset == 'yeast' else 9606


@ck.command()
@ck.option(
    '--train-data-file', '-trdf', default=f'data/data-train/{dataset_id}.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--valid-data-file', '-vldf', default=f'data/data-valid/{dataset_id}.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--test-data-file', '-tsdf', default=f'data/data-test/{dataset_id}.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/classPPIEmbed.pkl_last.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/relationPPIEmbed.pkl_last.pkl',
    help='Relation embedings file')
@ck.option(
    '--margin', '-m', default=-0.1,
    help='Loss margin')
def main(train_data_file, valid_data_file, test_data_file, cls_embeds_file, rel_embeds_file, margin):
    print('Evaluating')
    embedding_size = 50
    reg_norm = 1

    device = get_device()
    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    nb_relations = len(rel_df)
    print(f'#Classes: {nb_classes}, #Relations: {nb_relations}')

    embeds_list = cls_df['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df['classes'])}
    r_embeds_list = rel_df['embeddings'].values
    relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = torch.zeros((nb_classes, size), requires_grad=False).to(device)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = torch.from_numpy(emb).to(device)
    proteins = {}
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            proteins[k] = v
    offsets = torch.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]
    prot_index = list(proteins.values())
    prot_offsets = offsets[prot_index, :]
    prot_embeds = embeds[prot_index, :]
    prot_dict = {v: k for k, v in enumerate(prot_index)}

    # relations
    r_size = len(r_embeds_list[0])
    r_embeds = torch.zeros((nb_relations, r_size), requires_grad=False).to(device)
    for i, emb in enumerate(r_embeds_list):
        r_embeds[i, :] = torch.from_numpy(emb).to(device)

    print('Loading data')
    train_data = load_data(train_data_file, classes, relations)
    # valid_data = load_data(valid_data_file, classes, relations)
    train_labels = {}
    for c, r, d in train_data:
        c, r, d = prot_dict[classes[c]], relations[r], prot_dict[classes[d]]
        if r not in train_labels:
            train_labels[r] = torch.ones((len(prot_dict), len(prot_dict)), requires_grad=False).to(device)
        train_labels[r][c, d] = torch.inf

    test_data = load_data(test_data_file, classes, relations)

    top1 = 0.
    top10 = 0.
    top100 = 0.
    ftop1 = 0.
    ftop10 = 0.
    ftop100 = 0.
    ranks = torch.Tensor().to(device)
    franks = torch.Tensor().to(device)
    eval_data = test_data
    n = len(eval_data)

    batch_size = 1000
    num_batches = math.ceil(n / batch_size)
    eval_data = [(prot_dict[classes[c]], relations[r], prot_dict[classes[d]]) for c, r, d in eval_data]
    eval_data = torch.tensor(eval_data, requires_grad=False).to(device)
    r = eval_data[0, 1]
    assert ((eval_data[:, 1] == r).sum() == n)  # assume we use the same r everywhere

    for i in trange(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]

        batch_translated = prot_embeds[batch_data[:, 0]] + r_embeds[batch_data[:, 1]]
        batch_offsets = prot_offsets[batch_data[:, 0]]
        eucs = torch.abs(batch_translated[:, None, :] - torch.tile(prot_embeds, (current_batch_size, 1, 1)))
        dists = eucs - prot_offsets[None, :, :] + batch_offsets[:, None, :]
        dists = torch.linalg.norm(softplus(dists, beta=3), dim=2)  # TODO: dists.relu()

        index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
        batch_ranks = torch.take_along_dim(index, batch_data[:, 2].reshape(-1, 1), dim=1).flatten()

        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks = torch.cat((ranks, batch_ranks))

        dists = dists * train_labels[r.item()][batch_data[:, 0]]
        index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
        batch_ranks = torch.take_along_dim(index, batch_data[:, 2].reshape(-1, 1), dim=1).flatten()

        ftop1 += (batch_ranks <= 1).sum()
        ftop10 += (batch_ranks <= 10).sum()
        ftop100 += (batch_ranks <= 100).sum()
        franks = torch.cat((franks, batch_ranks))

    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank = torch.mean(ranks)
    ftop1 /= n
    ftop10 /= n
    ftop100 /= n
    fmean_rank = torch.mean(franks)

    ranks_dict = Counter(ranks.tolist())
    franks_dict = Counter(franks.tolist())
    rank_auc = compute_rank_roc(ranks_dict, len(proteins))
    frank_auc = compute_rank_roc(franks_dict, len(proteins))

    print(f'{dataset} {embedding_size} {margin} {reg_norm} {top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')
    print(f'{dataset} {embedding_size} {margin} {reg_norm} {ftop10:.2f} {ftop100:.2f} {fmean_rank:.2f} {frank_auc:.2f}')


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


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


def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0.0
    rmax = 0.0
    tmax = 0
    tpmax = 0
    fpmax = 0
    fnmax = 0
    for t in range(101):
        th = t / 100
        predictions = (preds >= th).astype(np.int32)
        tp = np.sum(labels & predictions)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            continue
        f = 2 * (p * r) / (p + r)
        if f > fmax:
            fmax = f
            pmax = p
            rmax = r
            tmax = t
            tpmax, fpmax, fnmax = tp, fp, fn
    return fmax, pmax, rmax, tmax, tpmax, fpmax, fnmax


@memory.cache
def load_data(data_file, classes, relations):
    data = []
    rel = f'<http://interacts>'
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            # data.append((id1, rel, id2))
            data.append((id2, rel, id1))
    return data


def is_inside(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst + rc <= rd


def is_intersect(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst <= rc + rd


def sim(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    overlap = max(0, (2 * rc - max(dst + rc - rd, 0)) / (2 * rc))
    edst = max(0, dst - rc - rd)
    res = (overlap + 1 / np.exp(edst)) / 2


if __name__ == '__main__':
    main()
