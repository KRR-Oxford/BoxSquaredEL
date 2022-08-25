import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu
from boxes import Boxes
import os
from model.loaded_models import ElbeLoadedModel


class ElbePlus(nn.Module):
    def __init__(self, device, class_, relation_num, embedding_dim, batch, margin=0, neg_dist=2,
                 ranking_fn='l2', num_neg=2, loss='mse'):
        super(ElbePlus, self).__init__()

        self.name = 'elbe+'
        self.margin = margin
        self.neg_dist = neg_dist
        self.class_num = len(class_)
        self.class_ = class_
        self.relation_num = relation_num
        self.device = device
        self.beta = None
        self.ranking_fn = ranking_fn
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.negative_sampling = True
        self.num_neg = num_neg

        self.classEmbeddingDict = self.init_embeddings(self.class_num, embedding_dim * 2)
        self.relationEmbeddingDict = self.init_embeddings(relation_num, embedding_dim)

    def init_embeddings(self, num_embeddings, dim, min=-1, max=1, normalise=True):
        embeddings = nn.Embedding(num_embeddings, dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        if normalise:
            embeddings.weight.data /= torch.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        return embeddings

    def get_boxes(self, embedding):
        return Boxes(embedding[:, :self.embedding_dim], torch.abs(embedding[:, self.embedding_dim:]))

    # boxes1 <= boxes2
    def inclusion_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.reshape(torch.linalg.norm(relu(diffs + boxes1.offsets - boxes2.offsets - self.margin), axis=1),
                             [-1, 1])
        return dist

    def disjoint_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.linalg.norm(relu(-diffs + boxes1.offsets + boxes2.offsets - self.margin), axis=1).reshape([-1, 1])
        return dist

    def neg_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.reshape(torch.linalg.norm(relu(diffs - boxes1.offsets - boxes2.offsets + self.margin), axis=1),
                             [-1, 1])
        return dist

    def nf1_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)
        return self.inclusion_loss(c_boxes, d_boxes)

    def nf2_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)
        e_boxes = self.get_boxes(e)

        intersection, lower, upper = c_boxes.intersect(d_boxes)
        return self.inclusion_loss(intersection, e_boxes) + torch.linalg.norm(relu(lower - upper), axis=1)

    def nf2_disjoint_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)
        return self.disjoint_loss(c_boxes, d_boxes)

    def nf3_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)

        return self.inclusion_loss(c_boxes.translate(r), d_boxes)

    def nf3_neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)

        return self.neg_loss(c_boxes.translate(r), d_boxes)

    def nf4_loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])
        r = self.relationEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 2])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)

        return self.inclusion_loss(c_boxes.translate(-r), d_boxes)

    def forward(self, input):
        batch = 512
        criterion = torch.nn.BCEWithLogitsLoss()

        rand_index = np.random.choice(len(input['nf1']), size=batch)
        nf1_data = input['nf1'][rand_index]
        nf1_data = nf1_data.to(self.device)
        loss1 = self.nf1_loss(nf1_data)
        if self.loss == 'mse':
            loss1 = loss1.square().mean()
        elif self.loss == 'bce':
            loss1 = criterion(-loss1, torch.ones_like(loss1))

        # nf2
        rand_index = np.random.choice(len(input['nf2']), size=batch)
        nf2_data = input['nf2'][rand_index]
        nf2_data = nf2_data.to(self.device)
        loss2 = self.nf2_loss(nf2_data)
        if self.loss == 'mse':
            loss2 = loss2.square().mean()
        elif self.loss == 'bce':
            loss2 = criterion(-loss2, torch.ones_like(loss2))

        # nf3
        rand_index = np.random.choice(len(input['nf3']), size=batch)
        nf3_data = input['nf3'][rand_index]
        nf3_data = nf3_data.to(self.device)
        loss3 = self.nf3_loss(nf3_data)
        if self.loss == 'mse':
            loss3 = loss3.square().mean()
        elif self.loss == 'bce':
            loss3 = criterion(-loss3, torch.ones_like(loss3))

        # nf4
        rand_index = np.random.choice(len(input['nf4']), size=batch)
        nf4_data = input['nf4'][rand_index]
        nf4_data = nf4_data.to(self.device)
        loss4 = self.nf4_loss(nf4_data)
        if self.loss == 'mse':
            loss4 = loss4.square().mean()
        elif self.loss == 'bce':
            loss4 = criterion(-loss4, torch.ones_like(loss4))

        # disJoint
        if len(input['disjoint']) == 0:
            disjoint_loss = 0
        else:
            rand_index = np.random.choice(len(input['disjoint']), size=batch)
            disjoint_data = input['disjoint'][rand_index]
            disjoint_data = disjoint_data.to(self.device)
            disjoint_loss = self.nf2_disjoint_loss(disjoint_data)
            if self.loss == 'mse':
                disjoint_loss = disjoint_loss.square().mean()
            elif self.loss == 'bce':
                disjoint_loss = criterion(-disjoint_loss, torch.zeros_like(disjoint_loss))

        rand_index = np.random.choice(len(input['nf3_neg0']), size=batch)
        neg_data = input['nf3_neg0'][rand_index]
        for i in range(1, self.num_neg):
            neg_data2 = input[f'nf3_neg{i}'][rand_index]
            neg_data = torch.cat([neg_data, neg_data2], dim=0)
        neg_data = neg_data.to(self.device)
        neg_loss = self.nf3_neg_loss(neg_data)
        if self.loss == 'mse':
            neg_loss = (self.neg_dist - neg_loss).square().mean()

        total_loss = [loss1 + loss2 + disjoint_loss + loss3 + loss4 + neg_loss]
        return total_loss

    def to_loaded_model(self):
        model = ElbeLoadedModel()
        model.embedding_size = self.embedding_dim
        model.class_embeds = self.classEmbeddingDict.weight.detach()
        model.relation_embeds = self.relationEmbeddingDict.weight.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''
        np.save(f'{folder}/class_embeds{suffix}.npy', self.classEmbeddingDict.weight.detach().cpu().numpy())
        np.save(f'{folder}/relations{suffix}.npy', self.relationEmbeddingDict.weight.detach().cpu().numpy())
