#!/usr/bin/env python
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model.BoxSquaredEL import BoxSquaredEL
from utils.ppi_data_loader import load_data, load_protein_data
from evaluate_ppi import compute_ranks, evaluate
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
    if len(sys.argv) > 1:
        sweep = sys.argv[1]
        wandb.agent(sweep_id=f'mathiasj/BoxSquaredEL/{sweep}', function=run)
    else:
        with open('configs.json', 'r') as f:
            configs = json.load(f)
        run(configs['PPI']['yeast'], use_wandb=True)


def run(config=None, use_wandb=True):
    if config is None:  # running a sweep
        num_epochs = 3500
        wandb.init()
    else:
        num_epochs = 5000
        mode = 'online' if use_wandb else 'disabled'
        wandb.init(mode=mode, project='BoxSquaredEL', entity='mathiasj', config=config)
    wandb.config['task'] = 'ppi'

    device = get_device()
    dataset = config['dataset']
    train_data, classes, proteins, relations = load_data(dataset)
    with open(f'data/PPI/{dataset}/classes.json', 'w+') as f:
        json.dump(classes, f)
    with open(f'data/PPI/{dataset}/proteins.json', 'w+') as f:
        json.dump(proteins, f)
    with open(f'data/PPI/{dataset}/relations.json', 'w+') as f:
        json.dump(relations, f)
    val_data = load_protein_data(dataset, 'val', proteins, relations)
    val_data = val_data[:1000]

    embedding_dim = 200
    num_neg = wandb.config.num_neg
    model = BoxSquaredEL(device, embedding_dim, len(classes), len(relations), len(proteins),
                         margin=wandb.config.margin, neg_dist=wandb.config.neg_dist,
                         reg_factor=wandb.config.reg_factor, num_neg=num_neg)
    wandb.config['model'] = model.name

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    if wandb.config.lr_schedule is None:
        scheduler = None
    else:
        scheduler = MultiStepLR(optimizer, milestones=[wandb.config.lr_schedule], gamma=0.1)
    model = model.to(device)
    out_folder = f'data/PPI/{dataset}/{model.name}'
    train(model, train_data, val_data, optimizer, scheduler, out_folder, num_neg, num_epochs=num_epochs)

    print('Computing test scores...')
    surrogate = evaluate(dataset, embedding_dim, split='val')
    wandb.log({'surrogate': surrogate})
    wandb.finish()


def train(model, train_data, val_data, optimizer, scheduler, out_folder, num_neg, num_epochs=5000,
          val_freq=100):
    model.train()
    wandb.watch(model)

    best_top10 = 0
    best_top100 = 0
    best_mr = sys.maxsize
    best_epoch = 0

    try:
        for epoch in trange(num_epochs):
            sample_negatives(train_data, num_neg)

            loss = model(train_data)
            if epoch % val_freq == 0 and val_data is not None:
                ranks, top1, top10, top100, filtered_ranks, ftop1, ftop10, ftop100 = \
                    compute_ranks(model.to_loaded_model(), val_data, model.device)
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

        ras = data['abox']['role_assertions']
        randoms = np.random.choice(data['prot_ids'], size=(ras.shape[0], 2))
        randoms = torch.from_numpy(randoms)
        new_tails = torch.cat([ras[:, [0, 1]], randoms[:, 0].reshape(-1, 1)], dim=1)
        new_heads = torch.cat([ras[:, [0]], randoms[:, 1].reshape(-1, 1), ras[:, [2]]], dim=1)
        new_neg = torch.cat([new_tails, new_heads], dim=0)
        data['abox'][f'role_assertions_neg{i}'] = new_neg


if __name__ == '__main__':
    main()
