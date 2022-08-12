from collections import Counter

import logging
import numpy as np
import torch
import math
from tqdm import trange, tqdm
from torch.nn.functional import softplus, relu

from ranking_result import RankingResult
from loaded_models import LoadedModel
from utils.utils import get_device
from utils.prediction_data_loader import load_data, load_test_data
from loaded_models import LoadedModel

logging.basicConfig(level=logging.INFO)


def main():
    evaluate('GALEN', 'prediction', model_name='boxsqel', embedding_size=200, ranking_fn='l2', beta=1, best=True)
    # evaluate('GO', embedding_size=200, ranking_fn='l1', beta=.5)
    # evaluate('ANATOMY', embedding_size=50, ranking_fn='l1', beta=.5)


def evaluate(dataset, task, model_name, embedding_size, beta, ranking_fn, best=True):
    device = get_device()

    model = LoadedModel.from_name(model_name, f'data/{dataset}/{task}/{model_name}', embedding_size, device, best)
    num_classes = model.class_embeds.shape[0]

    print('Loading data')
    _, classes, relations = load_data(dataset)
    assert (len(classes) == num_classes)
    test_data = load_test_data(dataset, classes)

    # acc = compute_accuracy(model.class_embeds, embedding_size, test_data, device)
    rankings = []
    for nf in ['nf1', 'nf2', 'nf3', 'nf4']:
        ranking = compute_ranks(model, test_data, num_classes, nf, device, ranking_fn, beta, use_tqdm=True)
        rankings.append(ranking)

    for i, nf in enumerate(['nf1', 'nf2', 'nf3', 'nf4']):
        print(nf.upper())
        print('=========')
        print(rankings[i])
        print()

    print('Combined')
    print('=========')
    print(combine_rankings(rankings, num_classes))


def combine_rankings(rankings, num_classes):
    combined_ranking = RankingResult(0, 0, 0, [], 0)
    for ranking in rankings:
        combined_ranking = combined_ranking.combine(ranking)
    ranks_dict = Counter(combined_ranking.ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    combined_ranking.auc = auc
    return combined_ranking


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


def compute_ranks(model, eval_data, num_classes, nf, device, ranking_fn, beta, batch_size=100, use_tqdm=False):
    if nf not in eval_data:
        raise ValueError('Tried to evaluate model on normal form not present in the evaluation data')
    eval_data = eval_data[nf]
    eval_data = eval_data.to(device)

    top1, top10, top100 = 0., 0., 0.
    ranks = []
    n = len(eval_data)
    num_batches = math.ceil(n / batch_size)

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]
        fun = f'compute_{nf}_ranks'
        if model.is_translational() and nf in ['nf3', 'nf4']:
            fun += '_translational'
        batch_ranks = globals()[fun](model, batch_data, current_batch_size)  # call the correct function based on NF
        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks += batch_ranks.tolist()

    ranks_dict = Counter(ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    return RankingResult(top1, top10, top100, ranks, auc)


def compute_nf1_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    batch_centers = centers[batch_data[:, 0]]

    dists = batch_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    dists[:, batch_data[:, 0]] = torch.inf  # TODO: finish filtering
    return dists_to_ranks(dists, batch_data[:, 1])


def compute_nf2_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    intersection, _, _ = c_boxes.intersect(d_boxes)
    dists = intersection.centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf3_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    bumps = model.bumps
    head_boxes = model.get_boxes(model.relation_heads)
    tail_boxes = model.get_boxes(model.relation_tails)

    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]
    d_bumps = bumps[batch_data[:, 2]]
    batch_heads = head_boxes.centers[batch_data[:, 1]]
    batch_tails = tail_boxes.centers[batch_data[:, 1]]

    bumped_c_centers = torch.tile(centers, (batch_size, 1, 1)) + d_bumps[:, None, :]
    bumped_d_centers = d_centers[:, None, :] + torch.tile(bumps, (batch_size, 1, 1))

    c_dists = bumped_c_centers - batch_heads[:, None, :]
    c_dists = torch.linalg.norm(c_dists, dim=2, ord=2)
    d_dists = bumped_d_centers - batch_tails[:, None, :]
    d_dists = torch.linalg.norm(d_dists, dim=2, ord=2)
    dists = c_dists + d_dists
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf4_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    bumps = model.bumps
    head_boxes = model.get_boxes(model.relation_heads)

    centers = class_boxes.centers
    c_bumps = bumps[batch_data[:, 1]]
    batch_heads = head_boxes.centers[batch_data[:, 0]]

    translated_heads = batch_heads - c_bumps
    dists = translated_heads[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf3_ranks_translational(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]
    batch_relations = model.relation_embeds[batch_data[:, 1]]

    translated_centers = d_centers - batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf4_ranks_translational(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_centers = centers[batch_data[:, 1]]
    batch_relations = model.relation_embeds[batch_data[:, 0]]

    translated_centers = c_centers - batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])


def dists_to_ranks(dists, targets):
    index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
    return torch.take_along_dim(index, targets.reshape(-1, 1), dim=1).flatten()


def compute_rank_roc(ranks, num_classes):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(num_classes)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / num_classes
    return auc


if __name__ == '__main__':
    main()
