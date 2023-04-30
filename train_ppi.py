#!/usr/bin/env python
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model.ElbePlus import ElbePlus
from model.Elbe import Elbe
from model.ELSoftplusBoxModel import ELSoftplusBoxModel
from model.BoxSquaredEL import BoxSquaredEL
from utils.ppi_data_loader import load_data, load_protein_data
from evaluate_ppi import compute_ranks, load_protein_index, evaluate
import logging
import torch
import numpy as np
from tqdm import trange
import wandb
import json
import sys

from utils.utils import get_device

logging.basicConfig(level=logging.INFO)


def main():
    torch.manual_seed(42)
    np.random.seed(12)
    run()


def run(use_wandb=False):
    dataset = 'yeast'

    if use_wandb:
        wandb.init(project='BoxSquaredEL', entity='mathiasj', config={'dataset': dataset, 'task': 'ppi'})
    else:
        wandb.init(mode='disabled')

    device = get_device()

    train_data, classes, proteins, relations = load_data(dataset)
    with open(f'data/PPI/{dataset}/classes.json', 'w+') as f:
        json.dump(classes, f)
    with open(f'data/PPI/{dataset}/proteins.json', 'w+') as f:
        json.dump(proteins, f)
    with open(f'data/PPI/{dataset}/relations.json', 'w+') as f:
        json.dump(relations, f)
    valid_data = load_protein_data(dataset, 'val', proteins, relations)

    embedding_dim = 200
    num_neg = 3
    model = BoxSquaredEL(device, embedding_dim, len(classes), len(relations), len(proteins), margin=0.05, neg_dist=3,
                         reg_factor=0.05, num_neg=num_neg)
    wandb.config['model'] = model.name

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # scheduler = MultiStepLR(optimizer, milestones=[2500], gamma=0.1)
    scheduler = None
    model = model.to(device)
    out_folder = f'data/PPI/{dataset}/{model.name}'
    train(model, train_data, valid_data, classes, optimizer, scheduler, out_folder, num_neg)

    print('Computing test scores...')
    evaluate(dataset, embedding_dim)


def train(model, train_data, val_data, classes, optimizer, scheduler, out_folder, num_neg, num_epochs=7000,
          val_freq=100):
    model.train()
    wandb.watch(model)

    prot_index, prot_dict = load_protein_index(classes)

    best_top10 = 0
    best_top100 = 0
    best_mr = sys.maxsize
    best_epoch = 0

    try:
        for epoch in trange(num_epochs):
            sample_negatives(train_data, num_neg)

            re = model(train_data)
            loss = sum(re)
            if epoch % 1000 == 0:
                print('epoch:', epoch, 'loss:', round(loss.item(), 3))
            if epoch % val_freq == 0 and val_data is not None:
                ranks, top1, top10, top100, franks, ftop1, ftop10, ftop100 = \
                    compute_ranks(model.to_loaded_model(), val_data[:1000], prot_index, prot_dict, model.device)
                ranks = ranks.cpu().numpy()
                wandb.log({'top10': top10, 'top100': top100, 'mean_rank': np.mean(ranks),
                           'median_rank': np.median(ranks)}, commit=False)
                # if ranking.top100 >= best_top100:
                if np.median(ranks) <= best_mr:
                    best_top10 = top10
                    best_top100 = top100
                    best_mr = np.median(ranks)
                    best_epoch = epoch
                    model.save(out_folder, best=True)
            wandb.log({'loss': loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    except KeyboardInterrupt:
        print('Interrupted. Stopping training...')

    wandb.finish()
    print(f'Best epoch: {best_epoch}')
    model.save(out_folder)


def sample_negatives(data, num_neg):
    for i in range(num_neg):
        nf3 = data['nf3']
        randoms = np.random.choice(data['prot_ids'], size=(nf3.shape[0], 2))
        randoms = torch.from_numpy(randoms)
        new_tails = torch.cat([nf3[:, [0, 1]], randoms[:, 0].reshape(-1, 1)], dim=1)
        new_heads = torch.cat([randoms[:, 1].reshape(-1, 1), nf3[:, [1, 2]]], dim=1)
        new_neg = torch.cat([new_tails, new_heads], dim=0)
        data[f'nf3_neg{i}'] = new_neg


if __name__ == '__main__':
    main()
