#!/usr/bin/env python
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model.ELBoxModel import ELBoxModel
from model.ElBallModel import ELBallModel
from model.ELSoftplusBoxModel import ELSoftplusBoxModel
from model.Original import Original
from model.BoxSqEL import BoxSqEL
from utils.prediction_data_loader import load_data, load_valid_data
import logging
import pandas as pd
from tqdm import trange
import wandb
from evaluate import compute_ranks, compute_accuracy, evaluate

from utils.utils import get_device
import sys

logging.basicConfig(level=logging.INFO)


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = 'GALEN'
    task = 'prediction'
    embedding_dim = 200

    wandb.init(project=f"{dataset}-{task}", entity="krr")

    device = get_device()

    # training procedure
    train_data, classes, relations = load_data(dataset)
    val_data = load_valid_data(dataset, classes)
    val_data['nf1'] = val_data['nf1'][:1000]
    # val_data = None
    print('Loaded data.')
    # model = Original(device, classes, len(relations), embedding_dim, batch=512, margin1=0.05, disjoint_dist=2)
    # model = ELBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=512, margin=0.05,
    #                    disjoint_dist=2, ranking_fn='l2')
    # model = ELSoftplusBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=512, margin=0,
    #                           beta=1, disjoint_dist=2, ranking_fn='softplus')
    # model = ELSoftplusBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=batch_size, margin=0,
    #                           beta=.5, disjoint_dist=5, ranking_fn='softplus')
    model = BoxSqEL(device, classes, len(relations), embedding_dim, batch=512, margin=0.05, disjoint_dist=2,
                    ranking_fn='l2', reg_factor=0.05)

    out_folder = f'data/{dataset}/{task}/{model.name}'

    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = MultiStepLR(optimizer, milestones=[2000], gamma=0.1)
    # scheduler = None
    model = model.to(device)
    train(model, train_data, val_data, optimizer, scheduler, out_folder, num_epochs=5000, val_freq=100)

    print('Computing test scores...')
    evaluate(dataset, task, model.name, embedding_size=model.embedding_dim, beta=model.beta,
             ranking_fn=model.ranking_fn, best=True)


def train(model, data, val_data, optimizer, scheduler, out_folder, num_epochs=2000, val_freq=100):
    model.train()
    wandb.watch(model)

    best_top10 = 0
    best_top100 = 0
    best_mr = sys.maxsize
    best_epoch = 0

    for epoch in trange(num_epochs):
        nf3 = data['nf3']
        randoms = np.random.choice(data['prot_ids'], size=(nf3.shape[0], 2))
        randoms = torch.from_numpy(randoms)
        new_tails = torch.cat([nf3[:, [0, 1]], randoms[:, 0].reshape(-1, 1)], dim=1)
        new_heads = torch.cat([randoms[:, 1].reshape(-1, 1), nf3[:, [1, 2]]], dim=1)
        new_neg = torch.cat([new_tails, new_heads], dim=0)
        data['nf3_neg'] = new_neg

        re = model(data)
        loss = sum(re)
        if epoch % 1000 == 0:
            print('epoch:', epoch, 'loss:', round(loss.item(), 3))
        if epoch % val_freq == 0 and val_data is not None:
            # acc = compute_accuracy(embeds, model.embedding_dim, val_data, model.device)
            # wandb.log({'acc': acc}, commit=False)
            ranking = compute_ranks(model.to_loaded_model(), val_data, 'nf1', model.device, model.ranking_fn,
                                    model.beta)
            wandb.log({'top10': ranking.top10, 'top100': ranking.top100, 'mean_rank': np.mean(ranking.ranks),
                       'median_rank': np.median(ranking.ranks)}, commit=False)
            # if ranking.top100 >= best_top100:
            if np.median(ranking.ranks) <= best_mr:
                best_top10 = ranking.top10
                best_top100 = ranking.top100
                best_mr = np.median(ranking.ranks)
                best_epoch = epoch
                model.save(out_folder, best=True)
        wandb.log({'loss': loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    wandb.finish()

    print(f'Best epoch: {best_epoch}')
    model.save(out_folder)


if __name__ == '__main__':
    main()
