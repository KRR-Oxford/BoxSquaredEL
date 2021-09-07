#!/usr/bin/env python
import click as ck
import numpy as np
import pandas as pd
from ELModel import ELModel
import tensorflow as tf
import logging
from tensorflow.keras import backend as K

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/train/4932.classes-normalized.owl',
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
    # SLURM JOB ARRAY INDEX
    pai = params_array_index
    if params_array_index != -1:
        orgs = ['9606', '4932']
        sizes = [50, 100, 200, 400]
        margins = [-0.1, -0.01, 0.0, 0.01, 0.1]
        reg_norms = [1, ]
        reg_norm = reg_norms[0]
        margin = margins[params_array_index % 5]
        params_array_index //= 5
        embedding_size = sizes[params_array_index % 4]
        params_array_index //= 4
        org = orgs[params_array_index % 2]
        print('Params:', org, embedding_size, margin, reg_norm)

        data_file = f'data/train/{org}.classes-normalized.owl'
        valid_data_file = f'data/valid/{org}.protein.links.v11.0.txt'
        out_classes_file = f'data/{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_cls.pkl'
        out_relations_file = f'data/{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_rel.pkl'
        loss_history_file = f'data/{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_loss.csv'
    train_data, classes, relations = load_data(data_file)
    valid_data = load_valid_data(valid_data_file, classes, relations)


'''load the normalized data(nf1, nf2, nf3, nf4)

Args: 
    filename: the normalized data, .owl format

Return:
    data: dictonary, nf1,nf2...data with triple or double class or relation index
    classes: dictonary, key is class name, value is according index
    relations: dictonary, key is relation name, value is according index
'''


def load_data(filename):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': []}
    with open(filename) as f:
        for line in f:
            # Ignore SubObjectPropertyOf(
            if line.startswith('SubObjectPropertyOf'):
                continue
            # Ignore SubClassOf(), totally 10 characters
            # remove subClassOf
            line = line.strip()[11:-1]
            if not line:
                continue

            if line.startswith('ObjectIntersectionOf('):  # And operation
                # C and D SubClassOf E

                # SubClassOf(ObjectIntersectionOf(<http://4932.YPL266W> <http://purl.obolibrary.org/obo/GO_0008150>) <http://purl.obolibrary.org/obo/GO_0060049>)
                it = line.split(' ')
                c = it[0][21:]  # <http://4932.YPL266W>
                d = it[1][:-1]  # <http://purl.obolibrary.org/obo/GO_0008150>
                e = it[2]  # <http://purl.obolibrary.org/obo/GO_0060049>
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if e not in classes:
                    classes[e] = len(classes)

                # C and D is subsetof E
                form = 'nf2'
                if e == 'owl:Nothing':
                    form = 'disjoint'

                # add the according index to data
                data[form].append((classes[c], classes[d], classes[e]))


            elif line.startswith('ObjectSomeValuesFrom('):
                # R some C SubClassOf D

                it = line.split(' ')
                r = it[0][21:]
                c = it[1][:-1]
                d = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf4'].append((relations[r], classes[c], classes[d]))
            elif line.find('ObjectSomeValuesFrom') != -1:
                # C SubClassOf R some D
                it = line.split(' ')
                c = it[0]
                r = it[1][21:]
                d = it[2][:-1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf3'].append((classes[c], relations[r], classes[d]))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf1'].append((classes[c], classes[d]))

    # Check if TOP in classes and insert if it is not there
    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
    if 'owl:Nothing' not in classes:
        classes['owl:Nothing'] = len(classes)

    prot_ids = []
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            prot_ids.append(v)
    prot_ids = np.array(prot_ids)

    # Add at least one disjointness axiom if there is 0
    if len(data['disjoint']) == 0:
        nothing = classes['owl:Nothing']
        n_prots = len(prot_ids)
        for i in range(10):
            it = np.random.choice(n_prots, 2)
            if it[0] != it[1]:
                data['disjoint'].append(
                    (prot_ids[it[0]], prot_ids[it[1]], nothing))
                break

    # Add corrupted triples for nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    inter_ind = 0
    for k, v in relations.items():
        if k == '<http://interacts>':
            inter_ind = v
    for c, r, d in data['nf3']:
        if r != inter_ind:
            continue
        data['nf3_neg'].append((c, r, np.random.choice(prot_ids)))
        data['nf3_neg'].append((np.random.choice(prot_ids), r, d))

    data['nf1'] = np.array(data['nf1'])
    data['nf2'] = np.array(data['nf2'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['disjoint'] = np.array(data['disjoint'])
    data['top'] = np.array([classes['owl:Thing'], ])
    data['nf3_neg'] = np.array(data['nf3_neg'])

    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]

    return data, classes, relations


'''load valid data

Args:
    valid_data_file: .txt file, one line means two interacted proteins
    classes: dictonary, key is class name, value is according index
    relations: dictonary, key is relation name, value is according index

Return value:
    data: classes[id1], relations[rel], classes[id2]
'''


def load_valid_data(valid_data_file, classes, relations):
    data = []
    rel = f'<http://interacts>'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data


if __name__ == '__main__':
    main()
