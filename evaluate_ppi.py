from collections import Counter

import json
import logging
from tqdm import trange
import numpy as np
import torch
from model.loaded_models import LoadedModel
from utils.ppi_data_loader import load_protein_data

import math
from utils.utils import get_device, memory

logging.basicConfig(level=logging.INFO)


def main():
    evaluate('yeast', 200)


def evaluate(dataset, embedding_size, split='test'):
    device = get_device()
    model = LoadedModel.from_name('boxsqel', f'data/PPI/{dataset}/boxsqel', embedding_size, device, best=True)
    with open(f'data/PPI/{dataset}/proteins.json', 'r') as f:
        proteins = json.load(f)
    with open(f'data/PPI/{dataset}/relations.json', 'r') as f:
        relations = json.load(f)

    print('Loading data')
    filtering_dict = load_filtering_dict(dataset, proteins, relations, device)
    test_data = load_protein_data(dataset, split, proteins, relations)

    ranks, top1, top10, top100, filtered_ranks, ftop1, ftop10, ftop100 = \
        compute_ranks(model, test_data, device, filtering_dict, use_tqdm=True)

    ranks_dict = Counter(ranks.tolist())
    franks_dict = Counter(filtered_ranks.tolist())
    rank_auc = compute_rank_roc(ranks_dict, len(proteins))
    frank_auc = compute_rank_roc(franks_dict, len(proteins))

    ranks = ranks.cpu().numpy()
    filtered_ranks = filtered_ranks.cpu().numpy()

    output = f'Standard: {top10:.2f},{top100:.2f},{np.mean(ranks):.2f},{np.median(ranks)},{rank_auc:.2f}\n' \
             f'Filtered: {ftop10:.2f},{ftop100:.2f},{np.mean(filtered_ranks):.2f},{np.median(filtered_ranks)},{frank_auc:.2f}'
    print(output)
    with open('output.txt', 'w+') as f:
        f.write(output)
    return np.median(filtered_ranks) - ftop100 - 0.1 * ftop10


def compute_ranks(model, eval_data, device, filtering_dict=None, use_tqdm=False):
    top1 = top10 = top100 = ftop1 = ftop10 = ftop100 = 0.
    ranks = torch.Tensor().to(device)
    filtered_ranks = torch.Tensor().to(device)
    num_eval_data = len(eval_data)

    batch_size = 100
    num_batches = math.ceil(num_eval_data / batch_size)
    eval_data = torch.tensor(eval_data, requires_grad=False).to(device)
    r = eval_data[0, 1]
    assert ((eval_data[:, 1] != r).sum() == 0)  # assume we use the same r everywhere

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, num_eval_data - start)
        batch_data = eval_data[start:start + current_batch_size, :]

        embeds = model.individual_embeds
        bumps = model.individual_bumps
        head_boxes = model.get_boxes(model.relation_heads)
        tail_boxes = model.get_boxes(model.relation_tails)

        d_embeds = embeds[batch_data[:, 2]]
        d_bumps = bumps[batch_data[:, 2]]
        batch_heads = head_boxes.centers[batch_data[:, 1]]
        batch_tails = tail_boxes.centers[batch_data[:, 1]]

        bumped_c_centers = torch.tile(embeds, (current_batch_size, 1, 1)) + d_bumps[:, None, :]
        bumped_d_centers = d_embeds[:, None, :] + torch.tile(bumps, (current_batch_size, 1, 1))

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

        if filtering_dict is not None:
            dists = dists * filtering_dict[r.item()][batch_data[:, 2]]
            index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
            batch_ranks = torch.take_along_dim(index, batch_data[:, 0].reshape(-1, 1), dim=1).flatten()

            ftop1 += (batch_ranks <= 1).sum()
            ftop10 += (batch_ranks <= 10).sum()
            ftop100 += (batch_ranks <= 100).sum()
            filtered_ranks = torch.cat((filtered_ranks, batch_ranks))

    top1 /= num_eval_data
    top10 /= num_eval_data
    top100 /= num_eval_data
    ftop1 /= num_eval_data
    ftop10 /= num_eval_data
    ftop100 /= num_eval_data

    return ranks, top1, top10, top100, filtered_ranks, ftop1, ftop10, ftop100


@memory.cache
def load_filtering_dict(dataset, proteins, relations, device):
    """Returns a dictionary structure that is used to compute filtered metrics."""
    train_data = load_protein_data(dataset, 'train', proteins, relations)
    filtering_dict = {}
    for c, r, d in train_data:
        if r not in filtering_dict:
            filtering_dict[r] = torch.ones((len(proteins), len(proteins)), requires_grad=False).to(device)
        filtering_dict[r][c, d] = torch.inf
    return filtering_dict


def compute_rank_roc(ranks, num_proteins):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(num_proteins)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / num_proteins
    return auc


if __name__ == '__main__':
    main()
