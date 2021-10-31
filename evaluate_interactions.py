#!/usr/bin/env python

import click as ck
import numpy
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque

import torch

from utils.utils import Ontology, FUNC_DICT

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)
epoch = '6000'
@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--train-data-file', '-trdf', default='data/data-train/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--valid-data-file', '-vldf', default='data/data-valid/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--test-data-file', '-tsdf', default='data/data-test/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/classEmbedPlot.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/relationEmbedPlot.pkl',
    help='Relation embedings file')
@ck.option(
    '--margin', '-m', default=-0.1,
    help='Loss margin')
@ck.option(
    '--params-array-index', '-pai', default=-1,
    help='Params array index')
def main(go_file, train_data_file, valid_data_file, test_data_file,
         cls_embeds_file, rel_embeds_file, margin, params_array_index):
    embedding_size = 50
    reg_norm = 1
    org = 'human'

    cls_df_tail = pd.read_pickle('data/classTailEmbedPlot.pkl')
    cls_df_head = pd.read_pickle('data/classHeadEmbedPlot.pkl')
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df_head)
    nb_relations = len(rel_df)

    embeds_list_tail = cls_df_tail['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df_tail['classes'])}
    rembeds_list = rel_df['embeddings'].values
    relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list_tail[0])
    embeds_tail = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list_tail):
        embeds_tail[i, :] = emb
    proteins_tail = {}
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            proteins_tail[k] = v
    rs = embeds_tail[:, embedding_size:]
    embeds_tail = embeds_tail[:, :embedding_size]
    prot_index_tail = list(proteins_tail.values())
    prot_rs_tail = rs[prot_index_tail, :]
    prot_embeds_tail = embeds_tail[prot_index_tail, :]
    prot_dict_tail = {v: k for k, v in enumerate(prot_index_tail)}

    #head
    embeds_list_head = cls_df_head['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df_head['classes'])}
    rembeds_list = rel_df['embeddings'].values
    relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list_head[0])
    embeds_head = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list_head):
        embeds_head[i, :] = emb
    proteins_head = {}
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            proteins_head[k] = v
    rs = embeds_head[:, embedding_size:]
    embeds_head = embeds_head[:, :embedding_size]
    prot_index_head = list(proteins_head.values())
    prot_rs_head = rs[prot_index_head, :]
    prot_embeds_head = embeds_head[prot_index_head, :]
    prot_dict_head = {v: k for k, v in enumerate(prot_index_head)}
    #####################################################################


    # relation
    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb

    train_data = load_data(train_data_file, classes, relations)
    valid_data = load_data(valid_data_file, classes, relations)
    trlabels = {}
    for c, r, d in train_data:
        c, r, d = prot_dict_head[classes[c]], relations[r], prot_dict_tail[classes[d]]
        if r not in trlabels:
            trlabels[r] = np.ones((len(prot_dict_head), len(prot_dict_head)), dtype=np.int32)
        trlabels[r][c, d] = 10000
    # for c, r, d in valid_data:
    #     c, r, d = prot_dict[classes[c]], relations[r], prot_dict[classes[d]]
    #     if r not in trlabels:
    #         trlabels[r] = np.ones((len(prot_embeds), len(prot_embeds)), dtype=np.int32)
    #     trlabels[r][c, d] = 1000

    test_data = load_data(test_data_file, classes, relations)
    # print(test_data_file, classes, relations)
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop1 = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    labels = {}
    preds = {}
    ranks = {}
    franks = {}
    eval_data = test_data
    n = len(eval_data)

    # model = torch.load('netPlot.pkl')
    # transfer_matrix_list = list(model.transfer_matrix.weight.clone().detach().cpu().numpy())

    transfer_matrix_embed = np.zeros((nb_relations, int(size * size / 4)), dtype=np.float32)
    # for i, emb in enumerate(transfer_matrix_list):
    #     transfer_matrix_embed[i, :] = emb

    with ck.progressbar(eval_data) as prog_data:
        for c, r, d in prog_data:
            c, r, d = prot_dict_head[classes[c]], relations[r], prot_dict_tail[classes[d]]
            if r not in labels:
                labels[r] = np.zeros((len(prot_dict_head), len(prot_dict_head)), dtype=np.int32)
            if r not in preds:
                preds[r] = np.zeros((len(prot_dict_head), len(prot_dict_head)), dtype=np.float32)
            labels[r][c, d] = 1

            # 蛋白质左下（原中心）

            ec = prot_embeds_head[c, :].reshape(1, -1)

            # 蛋白质右上（原半径）
            rc = prot_rs_head[c, :].reshape(1, -1)

            # relation左下
            er = rembeds[r, :].reshape(1, -1)

            ec += er
            rc += er

            prot_embedsNew = prot_embeds_tail

            prot_rsNew = prot_rs_tail
            # #1.圆的距离
            centerPro = (prot_embedsNew + prot_rsNew) / 2

            centerClass = (rc + ec) / 2

            # res = np.linalg.norm(centerPro - centerClass, axis=1) - np.linalg.norm(prot_embedsNew - prot_rsNew,
            #                                                                        axis=1) / 2 - np.linalg.norm(
            #     (rc - ec), axis=1) / 2

            #
            # 2.cosine距离 human 50 -0.1 1 0.10 0.55 244.48 0.96
            # dis1 = np.linalg.norm(prot_embedsNew,axis=1)+np.linalg.norm(ec,axis=1)
            # dis2 = np.linalg.norm(rc,axis=1)+np.linalg.norm(prot_rsNew,axis=1)
            # res = -np.sum(prot_embedsNew * ec,axis=1)/dis1 - np.sum(rc * prot_rsNew,axis=1)/dis2

            # 3.欧氏距离

            '''

            human 50 -0.1 1 0.09 0.56 267.52 0.95
            human 50 -0.1 1 0.24 0.74 219.65 0.96

            human 50 -0.1 1 0.08 0.45 445.89 0.92yeast-classes-normalized.owl
            human 50 -0.1 1 0.18 0.60 400.83 0.93
            '''

            res =     np.linalg.norm(prot_embedsNew  -ec, axis=1)
         #   print(res)

            '''
            human 50 -0.1 1 0.09 0.52 360.45 0.93
human 50 -0.1 1 0.23 0.68 313.47 0.94'''
            # 4.box 相交
            # startAll = np.maximum(prot_embedsNew, ec)
            # endAll = np.minimum(prot_rsNew, rc)
            # res = -np.sum(endAll-startAll,axis=1)#/(np.abs((np.sum(prot_rs-prot_embeds,axis=1))+np.abs(np.sum(rc-ec,axis=1)))+0.1)
            #  print(res)

            # #5.box 距离
            # startAll = np.maximum(prot_embedsNew -ec)
            # endAll = np.minimum(prot_rsNew, rc)
            # res = (np.sum(prot_embedsNew -ec,axis=1)+np.sum(prot_rsNew - rc,axis=1))#/(np.abs((np.sum(prot_rs-prot_embeds,axis=1))+np.abs(np.sum(rc-ec,axis=1)))+0.1)

            # #6 nf3loss
            # leftBottomLimit =  np.linalg.norm(np.maximum(prot_embedsNew - ec +0.01   , np.zeros(prot_rsNew.shape))   , axis=1)
            # righttopLimit =  np.linalg.norm(np.maximum(rc- prot_rsNew +0.01, np.zeros(prot_rsNew.shape)), axis=1)
            # res =  leftBottomLimit+righttopLimit
            #  print(res)

            # #圆心间距离
            # dst = np.linalg.norm(prot_embeds - ec.reshape(1, -1), axis=1)
            # dst = dst.reshape(-1, 1)(0, (2 * rc - np.maximum(dst + rc - prot_rs - margin, 0)) / (2 * rc))
            # else:
            # if rc > 0:
            #     overlap = np.maximum
            #     overlap = (np.maximum(dst - prot_rs - margin, 0) == 0).astype('float32')

            # edst = np.maximum(0, dst - rc - prot_rs - margin)
            # res = (overlap + 1 / np.exp(edst)) / 2

            # 圆心间距离 - 圆的半径和
            # 有交集就是0
            # res = np.maximum(0, dst - rc - prot_rs - margin)

            # res = np.sqrt(np.sum((edges*edges),axis=1))
            # print(res.shape,preds[r][c, :].shape)
            # for i in range(len(res)):
            #     preds[r][c][i]=res[i]
            # preds[r][c, :] = np.reshape(preds[r][c, :],[-1,1])
            # print(res.shape)
            preds[r][c, :] = res
            index = rankdata(res, method='average')

            rank = index[d]

            # print(rank,res[d])

            # print(1 / 0)
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            index = rankdata((res * trlabels[r][c, :]), method='average')
            rank = index[d]
            if rank == 1:
                ftop1 += 1
            if rank <= 10:
                ftop10 += 1
            if rank <= 100:
                ftop100 += 1
            fmean_rank += rank

            if rank not in franks:
                franks[rank] = 0
            franks[rank] += 1
        top1 /= n
        top10 /= n
        top100 /= n
        mean_rank /= n
        ftop1 /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n

    rank_auc = compute_rank_roc(ranks, len(proteins_head))
    frank_auc = compute_rank_roc(franks, len(proteins_head))

    print(f'{org} {embedding_size} {margin} {reg_norm} {top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')
    print(f'{org} {embedding_size} {margin} {reg_norm} {ftop10:.2f} {ftop100:.2f} {fmean_rank:.2f} {frank_auc:.2f}')


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc


def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0.0
    rmax = 0.0
    tmax = 0
    tpmax = 0
    fpmax = 0
    fnmax = 0
    for t in range(101):
        th = t / 100
        predictions = (preds >= th).astype(np.int32)
        tp = np.sum(labels & predictions)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            continue
        f = 2 * (p * r) / (p + r)
        if f > fmax:
            fmax = f
            pmax = p
            rmax = r
            tmax = t
            tpmax, fpmax, fnmax = tp, fp, fn
    return fmax, pmax, rmax, tmax, tpmax, fpmax, fnmax


def load_data(data_file, classes, relations):
    data = []
    rel = f'<http://interacts>'
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            # data.append((id1, rel, id2))
            data.append((id2, rel, id1))
    return data


def is_inside(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst + rc <= rd


def is_intersect(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst <= rc + rd


def sim(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    overlap = max(0, (2 * rc - max(dst + rc - rd, 0)) / (2 * rc))
    edst = max(0, dst - rc - rd)
    res = (overlap + 1 / np.exp(edst)) / 2


if __name__ == '__main__':
    main()