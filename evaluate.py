from collections import Counter

import logging
import numpy as np
import torch
import torch.nn.functional as F
import math
from tqdm import trange

from ranking_result import RankingResult
from utils.utils import get_device
from utils.data_loader import DataLoader
from model.loaded_models import LoadedModel, BoxELLoadedModel

logging.basicConfig(level=logging.INFO)


def main():
    evaluate('GALEN', 'prediction', model_name='boxsqel', embedding_size=200, best=True)


def evaluate(dataset, task, model_name, embedding_size, best=True, split='test'):
    device = get_device()

    model = LoadedModel.from_name(model_name, f'data/{dataset}/{task}/{model_name}', embedding_size, device, best)
    num_classes = model.class_embeds.shape[0] if model_name != 'boxel' else model.min_embedding.shape[0]

    print('Loading data')
    data_loader = DataLoader.from_task(task)
    _, classes, relations = data_loader.load_data(dataset)
    assert (len(classes) == num_classes)
    if split == 'test':
        test_data = data_loader.load_test_data(dataset, classes)
    elif split == 'val':
        test_data = data_loader.load_val_data(dataset, classes)
    else:
        raise ValueError('Unknown split.')

    nfs = ['nf1', 'nf2', 'nf3', 'nf4'] if task == 'prediction' else ['nf1']
    rankings = []
    for nf in nfs:
        ranking = compute_ranks(model, test_data, num_classes, nf, device, use_tqdm=True)
        rankings.append(ranking)

    output = '\n'.join([f'{nf.upper()}\n=========\n{rankings[i]}\n' for (i, nf) in enumerate(nfs)])
    if len(nfs) > 1:
        rankings.append(combine_rankings(rankings, num_classes))
        output += f'\nCombined\n=========\n{rankings[-1]}\n'

    print(output)
    with open('output.txt', 'w+') as f:
        f.write(output)

    with open('output_csv.txt', 'w+') as f:
        csv_output = '\n\n'.join([ranking.to_csv() for ranking in rankings])
        f.write(csv_output)

    return rankings


def combine_rankings(rankings, num_classes):
    combined_ranking = RankingResult(0, 0, 0, [], 0)
    for ranking in rankings:
        combined_ranking = combined_ranking.combine(ranking)
    ranks_dict = Counter(combined_ranking.ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    combined_ranking.auc = auc
    return combined_ranking


def compute_ranks(model, eval_data, num_classes, nf, device, batch_size=100, use_tqdm=False):
    if nf not in eval_data:
        raise ValueError('Tried to evaluate model on normal form not present in the evaluation data')
    eval_data = eval_data[nf]
    eval_data = eval_data.to(device)

    top1, top10, top100 = 0, 0, 0
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
        elif isinstance(model, BoxELLoadedModel):
            fun += '_boxel'

        batch_ranks = globals()[fun](model, batch_data, current_batch_size)  # call the correct function based on NF
        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks += batch_ranks.tolist()

    ranks_dict = Counter(ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    return RankingResult(top1.item(), top10.item(), top100.item(), ranks, auc)


def compute_nf1_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    batch_centers = centers[batch_data[:, 0]]

    dists = batch_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c <= c
    return dists_to_ranks(dists, batch_data[:, 1])


def compute_nf1_ranks_boxel(model, batch_data, batch_size):
    batch_mins = model.min_embedding[batch_data[:, 0]]
    batch_deltas = model.delta_embedding[batch_data[:, 0]]
    batch_maxs = batch_mins + torch.exp(batch_deltas)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

    inter_min = torch.max(batch_mins[:, None, :], all_mins)
    inter_max = torch.min(batch_maxs[:, None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)  # 100x23142
    dists = 1 - probs
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c <= c
    return dists_to_ranks(dists, batch_data[:, 1])


def compute_nf2_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    intersection, _, _ = c_boxes.intersect(d_boxes)
    dists = intersection.centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= c
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d
    return dists_to_ranks(dists, batch_data[:, 2])

def compute_nf2_ranks_boxel(model, batch_data, batch_size):
    c_mins = model.min_embedding[batch_data[:, 0]]
    c_deltas = model.delta_embedding[batch_data[:, 0]]
    c_maxs = c_mins + torch.exp(c_deltas)

    d_mins = model.min_embedding[batch_data[:, 1]]
    d_deltas = model.delta_embedding[batch_data[:, 1]]
    d_maxs = d_mins + torch.exp(d_deltas)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

    inter_min1 = torch.max(c_mins, d_mins)  # compute intersection between C and D
    inter_max1 = torch.min(c_maxs, d_maxs)

    inter_min = torch.max(inter_min1[:, None, :], all_mins)  # compute intersection between (C n D) and E
    inter_max = torch.min(inter_max1[:, None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)  # 100x23142
    dists = 1 - probs
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= c
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d
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


def compute_nf3_ranks_boxel(model, batch_data, batch_size):
    batch_mins = model.min_embedding[batch_data[:, 2]]
    batch_deltas = model.delta_embedding[batch_data[:, 2]]
    batch_maxs = batch_mins + torch.exp(batch_deltas)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))
    relations = model.relation_embedding[batch_data[:, 1]]
    scalings = model.scaling_embedding[batch_data[:, 1]]
    translated_mins = all_mins * (scalings[:, None, :] + 1e-8) + relations[:, None, :]
    translated_maxs = all_maxs * (scalings[:, None, :] + 1e-8) + relations[:, None, :]

    inter_min = torch.max(batch_mins[:, None, :], translated_mins)
    inter_max = torch.min(batch_maxs[:, None, :], translated_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    batch_volumes = F.softplus(translated_maxs - translated_mins).prod(2)
    log_box2 = torch.log(torch.clamp(batch_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection - log_box2)  # 100x23142
    dists = 1 - probs
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


def compute_nf4_ranks_boxel(model, batch_data, batch_size):
    batch_mins = model.min_embedding[batch_data[:, 1]]
    batch_deltas = model.delta_embedding[batch_data[:, 1]]
    batch_maxs = batch_mins + torch.exp(batch_deltas)
    relations = model.relation_embedding[batch_data[:, 0]]
    scalings = model.scaling_embedding[batch_data[:, 0]]
    translated_mins = (batch_mins - relations) / (scalings + 1e-8)
    translated_maxs = (batch_maxs - relations) / (scalings + 1e-8)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

    inter_min = torch.max(translated_mins[:, None, :], all_mins)
    inter_max = torch.min(translated_maxs[:, None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)  # 100x23142
    dists = 1 - probs
    return dists_to_ranks(dists, batch_data[:, 0])


def dists_to_ranks(dists, targets):
    index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
    return torch.take_along_dim(index, targets.reshape(-1, 1), dim=1).flatten()


def compute_rank_roc(ranks, num_classes):
    sorted_ranks = sorted(list(ranks.keys()))
    tprs = [0]
    fprs = [0]
    tpr = 0
    num_triples = sum(ranks.values())
    num_negatives = (num_classes - 1) * num_triples
    for x in sorted_ranks:
        tpr += ranks[x]
        tprs.append(tpr / num_triples)
        fp = sum([(x - 1) * v if k <= x else x * v for k, v in ranks.items()])
        fprs.append(fp / num_negatives)

    tprs.append(1)
    fprs.append(1)
    auc = np.trapz(tprs, fprs)
    return auc


if __name__ == '__main__':
    main()
