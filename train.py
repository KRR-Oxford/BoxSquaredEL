#!/usr/bin/env python
import click as ck
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model.ELBoxModel import ELBoxModel
from model.ElBallModel import ELBallModel
from model.ELSoftplusBoxModel import ELSoftplusBoxModel
from model.Original import Original
from model.ELBox2 import ELBox2
from utils.el_data_loader import load_data, load_valid_data
import logging
import pandas as pd
from tqdm import trange
import wandb
from evaluate import compute_ranks, compute_accuracy, evaluate

from utils.utils import get_device
import sys

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--batch-size', '-bs', default=512,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1000,
    help='Training epochs')
@ck.option(
    '--device', '-d', default='gpu:0',
    help='GPU Device ID')
@ck.option(
    '--embedding-size', '-es', default=50,
    help='Embeddings size')
@ck.option(
    '--reg-norm', '-rn', default=1,
    help='Regularization norm')
@ck.option(
    '--margin', '-m', default=-0.1,
    help='Loss margin')
@ck.option(
    '--learning-rate', '-lr', default=0.01,
    help='Learning rate')
@ck.option(
    '--params-array-index', '-pai', default=-1,
    help='Params array index')
@ck.option(
    '--loss-history-file', '-lhf', default='data/loss_history.csv',
    help='Pandas pkl file with loss history')
def main(batch_size, epochs, device, embedding_size, reg_norm, margin,
         learning_rate, params_array_index, loss_history_file):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = 'GALEN'
    embedding_dim = 200
    out_classes_file = f'data/{dataset}/classELEmbed'
    out_relations_file = f'data/{dataset}/relationELEmbed'
    val_file = f'data/{dataset}/{dataset}_valid.txt'

    wandb.init(project=f"el2box-{dataset}-boxe", entity="krr")

    device = get_device()

    # training procedure
    train_data, classes, relations = load_data(dataset)
    val_data = load_valid_data(val_file, classes, relations)
    print('Loaded data.')
    # model = Original(device, classes, len(relations), embedding_dim, batch_size, margin1=0.1)
    # model = ELBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=batch_size, margin=0.1,
    #                    disjoint_dist=2, ranking_fn='l1')
    # model = ELSoftplusBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=batch_size, margin=0,
    #                           beta=1, disjoint_dist=2, ranking_fn='softplus')
    # model = ELSoftplusBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=batch_size, margin=0,
    #                           beta=.5, disjoint_dist=5, ranking_fn='softplus')
    model = ELBox2(device, classes, len(relations), embedding_dim, batch_size, margin=0.05, disjoint_dist=2,
                   ranking_fn='l2', reg_factor=0.1)

    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = MultiStepLR(optimizer, milestones=[2000], gamma=0.1)
    # scheduler = None
    model = model.to(device)
    train(model, train_data, val_data, optimizer, scheduler, out_classes_file, out_relations_file, classes, relations,
          num_epochs=5000, val_freq=100)

    print('Computing test scores...')
    evaluate(dataset, embedding_size=model.embedding_dim, beta=model.beta, ranking_fn=model.ranking_fn, last=True)


def train(model, data, val_data, optimizer, scheduler, out_classes_file, out_relations_file, classes, relations,
          num_epochs=2000,
          val_freq=100):
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
        wandb.log({'loss': loss})
        if epoch % 1000 == 0:
            print('epoch:', epoch, 'loss:', round(loss.item(), 3))
        if epoch % val_freq == 0:
            embeds = model.classEmbeddingDict.weight.clone().detach()
            acc = compute_accuracy(embeds, model.embedding_dim, val_data, model.device)
            wandb.log({'acc': acc})
            ranking = compute_ranks(embeds, model.embedding_dim, val_data[:1000], model.device, model.ranking_fn,
                                    model.beta)
            wandb.log({'top10': ranking.top10, 'top100': ranking.top100, 'mean_rank': np.mean(ranking.ranks),
                       'median_rank': np.median(ranking.ranks)})
            if ranking.top100 >= best_top100:
                # if np.mean(ranking.ranks) <= best_mr:
                best_top10 = ranking.top10
                best_top100 = ranking.top100
                best_mr = np.mean(ranking.ranks)
                best_epoch = epoch
                save_model(model, f'{out_classes_file}_best.pkl', f'{out_relations_file}_best.pkl', classes, relations)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    wandb.finish()

    print(f'Best epoch: {best_epoch}')
    save_model(model, f'{out_classes_file}_last.pkl', f'{out_relations_file}_last.pkl', classes, relations)


def save_model(model, cls_file, rel_file, classes, relations):
    df = pd.DataFrame(
        {'classes': list(classes.keys()),
         'embeddings': list(model.classEmbeddingDict.weight.clone().detach().cpu().numpy())})
    df.to_pickle(cls_file)

    # df = pd.DataFrame(
    #     {'relations': list(relations.keys()),
    #      'embeddings': list(model.relationEmbeddingDict.weight.clone().detach().cpu().numpy())})
    # df.to_pickle(rel_file)


if __name__ == '__main__':
    main()
