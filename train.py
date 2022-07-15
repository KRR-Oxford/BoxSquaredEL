#!/usr/bin/env python
import click as ck
import torch.optim as optim
from model.ELBoxlModel import ELBoxModel
from utils.el_data_loader import load_data, load_valid_data
import logging
import pandas as pd
from tqdm import trange
import wandb
from evaluate import compute_ranks, compute_accuracy

from utils.utils import get_device

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
    dataset = 'ANATOMY'
    embedding_dim = 50
    out_classes_file = f'data/{dataset}/classELEmbed'
    out_relations_file = f'data/{dataset}/relationELEmbed'
    val_file = f'data/{dataset}/{dataset}_valid.txt'

    wandb.init(project=f"el2box-{dataset}", entity="krr")

    device = get_device()

    # training procedure
    train_data, classes, relations = load_data(dataset)
    val_data = load_valid_data(val_file, classes, relations)
    print('Loaded data.')
    model = ELBoxModel(device, classes, len(relations), embedding_dim=embedding_dim, batch=batch_size, margin1=-0.1)  # TODO: margin


    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    model = model.to(device)
    train(model, train_data, val_data, optimizer, out_classes_file, out_relations_file, classes, relations, val_freq=100)
    # model.eval()

    # model = model.to(device)
    # save_model(model, out_classes_file, out_relations_file, classes, relations)


def train(model, data, val_data, optimizer, out_classes_file, out_relations_file, classes, relations, num_epochs=7000, val_freq=100):
    model.train()
    wandb.watch(model)
    best_top100 = 0
    best_epoch = 0

    for epoch in trange(num_epochs):
        re = model(data)
        loss = sum(re)
        wandb.log({'loss': loss})
        if epoch % 1000 == 0:
            print('epoch:', epoch, 'loss:', round(loss.item(), 3))
        if epoch % val_freq == 0:
            embeds = model.classEmbeddingDict.weight.clone().detach()
            acc = compute_accuracy(embeds, model.embedding_dim, val_data, model.device)
            wandb.log({'acc': acc})
            top1, top10, top100, mean_rank, ranks = compute_ranks(embeds, model.embedding_dim, val_data[:1000], model.device)
            wandb.log({'top10': top10, 'top100': top100, 'mean_rank': mean_rank})
            if top100 > best_top100:
                best_top100 = top100
                best_epoch = epoch
                save_model(model, f'{out_classes_file}_best.pkl', f'{out_relations_file}_best.pkl', classes, relations)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Best epoch: {best_epoch}')


def save_model(model, cls_file, rel_file, classes, relations):
    df = pd.DataFrame(
        {'classes': list(classes.keys()),
         'embeddings': list(model.classEmbeddingDict.weight.clone().detach().cpu().numpy())})
    df.to_pickle(cls_file)

    df = pd.DataFrame(
        {'relations': list(relations.keys()),
         'embeddings': list(model.relationEmbeddingDict.weight.clone().detach().cpu().numpy())})
    df.to_pickle(rel_file)


if __name__ == '__main__':
    main()
