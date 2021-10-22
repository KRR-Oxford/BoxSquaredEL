#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt

from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--cls-embeds-file', '-cef', default='../data/classEmbedPlot.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='../data/relationEmbedPlot.pkl',
    help='Relation embedings file')
@ck.option(
    '--epoch', '-e', default='',
    help='Epoch embeddings')
def main(go_file, cls_embeds_file, rel_embeds_file, epoch):

    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    nb_relations = len(rel_df)
    embeds_list = cls_df['embeddings'].values
    classes = {k: v for k, v in enumerate(cls_df['classes'])}
    rembeds_list = rel_df['embeddings'].values
    relations = {k: v for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    l1 = embeds[:, :-2]
    r1 = embeds[:, 2:]

    embeds = (l1+r1)/2
    rs = np.linalg.norm(l1-r1,axis =1)
   # print(embeds)

    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb

    plot_embeddings(l1,r1, classes, epoch)

def plot_embeddings(left,right, classes, epoch):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # if embeds.shape[1] > 2:
    #     embeds = TSNE().fit_transform(embeds)

    fig, ax =  plt.subplots()
   # plt.axis('equal')
    ax.set_xlim(-5, 4)
    ax.set_ylim(-3, 4)
    for i in range(left.shape[0]):
        # if classes[i].startswith('owl:'):
        #     continue
        x, y = left[i, 0], left[i, 1]
        width= right[i, 0]-x
        height = right[i, 1]-y

        ax.add_artist(plt.Rectangle(
            (x, y), width,height, fill=False, edgecolor=colors[i % len(colors)], label=classes[i],linewidth=1))
        ax.annotate(classes[i], xy=(x+width/2, y+height/2), fontsize=3, ha="center", color=colors[i % len(colors)])
    # ax.legend()
    ax.grid(True)
    # filename = 'embeds.svg'
    # if epoch:
    filename = 'embeds_'+str(epoch)+'.svg'
    plt.savefig(filename)
    # plt.plot([0.01,0.01],[0.01,0.01])
    # plt.show()


if __name__ == '__main__':
   main()
   # plt.plot([0.01, 0.01], [0.01, 0.01],'*')
   # plt.savefig('embeds.pdf')
