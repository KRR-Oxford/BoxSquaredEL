import click as ck
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)


@ck.command()
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
    '--cls-embeds-file', '-cef', default='data/classPPIEmbed.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/relationPPIEmbed.pkl',
    help='Relation embedings file')
@ck.option(
    '--margin', '-m', default=-0.1,
    help='Loss margin')
def main(train_data_file, valid_data_file, test_data_file, cls_embeds_file, rel_embeds_file, margin):
    print('Evaluating')
    embedding_size = 50
    reg_norm = 1
    org = 'yeast'

    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    nb_relations = len(rel_df)
    print(f'#Classes: {nb_classes}, #Relations: {nb_relations}')

    embeds_list = cls_df['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df['classes'])}
    r_embeds_list = rel_df['embeddings'].values
    relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    proteins = {}
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            proteins[k] = v
    offsets = embeds[:, embedding_size:]
    embeds = embeds[:, :embedding_size]
    prot_index = list(proteins.values())
    prot_offsets = offsets[prot_index, :]
    prot_embeds = embeds[prot_index, :]
    prot_dict = {v: k for k, v in enumerate(prot_index)}

    # relations
    r_size = len(r_embeds_list[0])
    r_embeds = np.zeros((nb_relations, r_size), dtype=np.float32)
    for i, emb in enumerate(r_embeds_list):
        r_embeds[i, :] = emb

    print('Loading data')
    train_data = load_data(train_data_file, classes, relations)
    # valid_data = load_data(valid_data_file, classes, relations)
    train_labels = {}
    for c, r, d in train_data:
        c, r, d = prot_dict[classes[c]], relations[r], prot_dict[classes[d]]
        if r not in train_labels:
            train_labels[r] = np.ones((len(prot_dict), len(prot_dict)), dtype=np.float32)
        train_labels[r][c, d] = np.inf

    test_data = load_data(test_data_file, classes, relations)

    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop1 = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    ranks = {}
    franks = {}
    eval_data = test_data
    n = len(eval_data)

    for c, r, d in tqdm(eval_data, total=len(eval_data)):
        c, r, d = prot_dict[classes[c]], relations[r], prot_dict[classes[d]]

        embedding = prot_embeds[c, :].reshape(1, -1)
        offset = np.abs(prot_offsets[c, :].reshape(1, -1))
        relation = r_embeds[r, :].reshape(1, -1)

        prot_embeds_new = prot_embeds
        prot_offsets_new = np.abs(prot_offsets)

        euc = np.abs(embedding + relation - prot_embeds_new)
        maximum = np.maximum(euc - prot_offsets_new + offset, np.zeros(euc.shape))
        res = np.reshape((np.linalg.norm(maximum, axis=1)), -1)
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
        index = rankdata((res * train_labels[r][c, :]), method='average')
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

    rank_auc = compute_rank_roc(ranks, len(proteins))
    frank_auc = compute_rank_roc(franks, len(proteins))

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
