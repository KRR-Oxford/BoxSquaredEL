#!/usr/bin/env python
import click as ck
import torch.optim as optim
from ELModel import ELModel
from utils.elDataLoader import load_data, load_valid_data
import logging
import torch

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/data-train/yeast-classes-normalized.owl',
    help='Normalized ontology file (Normalizer.groovy)')
@ck.option(
    '--valid-data-file', '-vdf', default='data/valid/4932.protein.links.v11.0.txt',
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
    print(torch.cuda.is_available())

    train_data, classes, relations = load_data(data_file)
    model = ELModel(device,len(classes), len(relations), embedding_dim=50)
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    train(model,train_data, optimizer)
    model = model.to(device)



def train(model, data, optimizer, num_epochs=200):
    for epoch in range(num_epochs):
        model.train()
        loss = model(data)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




if __name__ == '__main__':
    main()
