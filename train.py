#!/usr/bin/env python
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model.Elem import Elem
from model.EmELpp import EmELpp
from model.Elbe import Elbe
from model.BoxEL import BoxEL
from model.BoxSquaredEL import BoxSquaredEL
from utils.data_loader import DataLoader
import logging
from tqdm import trange
import wandb
from evaluate import compute_ranks, evaluate

from utils.utils import get_device
import sys

logging.basicConfig(level=logging.INFO)


def main():
    torch.manual_seed(42)
    np.random.seed(12)
    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        count = None if len(sys.argv) <= 2 else sys.argv[2]
        print(count)
        wandb.agent(sweep_id=f'mathiasj/el-baselines/{sweep_id}', function=run, count=count)
    else:
        with open('configs.json', 'r') as f:
            configs = json.load(f)
        run(config=configs['GALEN']['prediction'], use_wandb=True)


def run(config=None, use_wandb=True, split='val'):
    if config is None:  # running a sweep
        num_epochs = 5000
        wandb.init()
    else:
        num_epochs = 5000 if 'epochs' not in config else config['epochs']
        mode = 'online' if use_wandb else 'disabled'
        wandb.init(mode=mode, project='BoxSquaredEL', entity='mathiasj', config=config)

    embedding_dim = 200
    num_neg = wandb.config.num_neg if 'num_neg' in wandb.config else 1
    dataset = wandb.config.dataset
    task = wandb.config.task

    device = get_device()
    data_loader = DataLoader.from_task(task)
    train_data, classes, relations = data_loader.load_data(dataset)
    val_data = data_loader.load_val_data(dataset, classes)
    val_data['nf1'] = val_data['nf1'][:1000]
    print('Loaded data.')
    # model = Elem(device, classes, len(relations), embedding_dim, margin=0.00)
    # model = EmELpp(device, classes, len(relations), embedding_dim, margin=0.05)
    # model = Elbe(device, classes, len(relations), embedding_dim, margin=0.05)
    # model = BoxEL(device, classes, len(relations), embedding_dim)
    # model = ElbePlus(device, classes, len(relations), embedding_dim=embedding_dim, margin=0.05, neg_dist=2,
    #                  num_neg=num_neg)
    model = BoxSquaredEL(device, embedding_dim, len(classes), len(relations),
                         margin=wandb.config.margin, neg_dist=wandb.config.neg_dist,
                         reg_factor=wandb.config.reg_factor, num_neg=num_neg)
    wandb.config['model'] = model.name

    out_folder = f'data/{dataset}/{task}/{model.name}'

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    if wandb.config.lr_schedule is None:
        scheduler = None
    else:
        scheduler = MultiStepLR(optimizer, milestones=[wandb.config.lr_schedule], gamma=0.1)
    model = model.to(device)

    if not model.negative_sampling and task != 'old':
        sample_negatives(train_data, 1)

    train(model, train_data, val_data, len(classes), optimizer, scheduler, out_folder, num_neg, num_epochs=num_epochs,
          val_freq=100)

    print('Computing test scores...')
    scores = evaluate(dataset, task, model.name, embedding_size=model.embedding_dim, best=True, split=split)
    combined_scores = scores[-1]
    surrogate = np.median(combined_scores.ranks) - combined_scores.top100 / len(combined_scores) - \
                0.1 * combined_scores.top10 / len(combined_scores)
    wandb.log({'surrogate': surrogate})
    wandb.finish()
    return scores


def train(model, data, val_data, num_classes, optimizer, scheduler, out_folder, num_neg, num_epochs=2000, val_freq=100):
    model.train()
    wandb.watch(model)

    best_top10 = 0
    best_top100 = 0
    best_median = sys.maxsize
    best_mean = sys.maxsize
    best_epoch = 0

    try:
        for epoch in trange(num_epochs):
            if model.negative_sampling:
                sample_negatives(data, num_neg)

            loss = model(data)
            if epoch % val_freq == 0 and val_data is not None:
                ranking = compute_ranks(model.to_loaded_model(), val_data, num_classes, 'nf1', model.device)
                wandb.log({'top10': ranking.top10 / len(ranking), 'top100': ranking.top100 / len(ranking),
                           'mean_rank': np.mean(ranking.ranks), 'median_rank': np.median(ranking.ranks)}, commit=False)
                # if ranking.top100 >= best_top100:
                if np.median(ranking.ranks) <= best_median:
                    # if np.mean(ranking.ranks) <= best_mean:
                    best_top10 = ranking.top10
                    best_top100 = ranking.top100
                    best_median = np.median(ranking.ranks)
                    best_mean = np.mean(ranking.ranks)
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

    print(f'Best epoch: {best_epoch}')
    model.save(out_folder)


def sample_negatives(data, num_neg):
    for i in range(num_neg):
        nf3 = data['nf3']
        randoms = np.random.choice(data['class_ids'], size=(nf3.shape[0], 2))
        randoms = torch.from_numpy(randoms)
        new_tails = torch.cat([nf3[:, [0, 1]], randoms[:, 0].reshape(-1, 1)], dim=1)
        new_heads = torch.cat([randoms[:, 1].reshape(-1, 1), nf3[:, [1, 2]]], dim=1)
        new_neg = torch.cat([new_tails, new_heads], dim=0)
        data[f'nf3_neg{i}'] = new_neg


if __name__ == '__main__':
    main()
