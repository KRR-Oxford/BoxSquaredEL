#!/usr/bin/env python
import click as ck
import numpy as np
import torch.optim as optim
from model.ElBallModel import  ELBallModel
from utils.elDataLoader import load_data, load_valid_data
import logging
import torch

from utils.plotBall_embedding import plot_embeddings

logging.basicConfig(level=logging.INFO)
import pandas as pd

@ck.command()
#family_normalized.owl
#yeast-classes-normalized.owl
@ck.option(
    '--data-file', '-df', default='data/data-train/go_latest_norm_mod.owl',
    help='Normalized ontology file (Normalizer.groovy)')
@ck.option(
    '--valid-data-file', '-vdf', default='data/valid/4932.protein.links.v10.5.txt',
    help='Validation data set')
@ck.option(
    '--out-classes-file', '-ocf', default='data/cls_embeddings.pkl',
    help='Pandas pkl file with class embeddings')
@ck.option(
    '--out-relations-file', '-orf', default='data/rel_embeddings.pkl',
    help='Pandas pkl file with relation embeddings')
@ck.option(
    '--batch-size', '-bs', default=256,
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
def main(data_file, valid_data_file, out_classes_file, out_relations_file,
         batch_size, epochs, device, embedding_size, reg_norm, margin,
         learning_rate, params_array_index, loss_history_file):
    device = torch.device('cuda:0')

    #training procedure
    train_data, classes, relations = load_data(data_file)
    model = ELBallModel(device,len(classes), len(relations), embedding_dim=50, margin=0)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    model = model.to(device)
    train(model,train_data,optimizer,classes )
    model.eval()


    cls_file = 'data/ballClassEmbed.pkl'
    rel_file = 'data/ballRelationEmbed.pkl'

    df = pd.DataFrame(
        {'classes': list(classes.keys()), 'embeddings': list(model.classEmbeddingDict.weight.clone().detach().cpu().numpy())})
    df.to_pickle(cls_file)

    df = pd.DataFrame(
        {'relations': list(relations.keys()), 'embeddings': list(model.relationEmbeddingDict.weight.clone().detach().cpu().numpy())})
    df.to_pickle(rel_file)

    #store embedding





#ballRelationEmbed

def train(model, data, optimizer, aclasses,num_epochs=2000):
    model.train()
    for epoch in range(num_epochs):
        #model.zero_grad()
        loss = model(data)
        print('epoch:',epoch,'loss:',round(loss.item(),3))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # if epoch%100==0:
        #
        #     cls_df = list(model.classEmbeddingDict.weight.clone().detach().cpu().numpy())
        #
        #     nb_classes = len(cls_df)
        #
        #     embeds_list = cls_df
        #     classes = {k: v for k, v in enumerate(aclasses)}
        #
        #
        #     size = len(embeds_list[0])
        #     embeds = np.zeros((nb_classes, size), dtype=np.float32)
        #     for i, emb in enumerate(embeds_list):
        #         embeds[i, :] = emb
        #     rs = np.abs(embeds[:, -1])
        #
        #
        #     embeds = embeds[:, :-1]
        #
        #
        #     plot_embeddings(embeds, rs, classes, epoch)

if __name__ == '__main__':
    main()