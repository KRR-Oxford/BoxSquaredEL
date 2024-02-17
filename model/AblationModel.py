import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu
from boxes import Boxes
import os
from model.loaded_models import ElbeLoadedModel


class AblationModel(nn.Module):
    """
    A version of Box^2EL in which roles are represented via TransE for ablation study E.1.
    """
    def __init__(self, device, embedding_dim, num_classes, num_roles, margin=0, neg_dist=2, num_neg=2, batch_size=512):
        super(AblationModel, self).__init__()

        self.name = 'ablation'
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_roles = num_roles
        self.margin = margin
        self.neg_dist = neg_dist
        self.num_neg = num_neg
        self.batch_size = batch_size

        self.negative_sampling = True

        self.class_embeds = self.init_embeddings(self.num_classes, embedding_dim * 2)
        self.relation_embeds = self.init_embeddings(num_roles, embedding_dim)

    def init_embeddings(self, num_embeddings, dim, min=-1, max=1, normalise=True):
        if num_embeddings == 0:
            return None
        embeddings = nn.Embedding(num_embeddings, dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        if normalise:
            embeddings.weight.data /= torch.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        return embeddings

    def get_boxes(self, embedding):
        return Boxes(embedding[:, :self.embedding_dim], torch.abs(embedding[:, self.embedding_dim:]))

    def get_class_boxes(self, nf_data, *indices):
        return (self.get_boxes(self.class_embeds(nf_data[:, i])) for i in indices)

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

    def nf1_loss(self, nf1_data):
        c_boxes, d_boxes = self.get_class_boxes(nf1_data, 0, 1)
        return self.inclusion_loss(c_boxes, d_boxes)

    def nf2_loss(self, nf2_data):
        c_boxes, d_boxes, e_boxes = self.get_class_boxes(nf2_data, 0, 1, 2)
        intersection, lower, upper = c_boxes.intersect(d_boxes)
        return self.inclusion_loss(intersection, e_boxes) + torch.linalg.norm(relu(lower - upper), axis=1)

    def nf2_disjoint_loss(self, disjoint_data):
        c_boxes, d_boxes = self.get_class_boxes(disjoint_data, 0, 1)
        return self.disjoint_loss(c_boxes, d_boxes)

    def nf3_loss(self, nf3_data):
        c_boxes, d_boxes = self.get_class_boxes(nf3_data, 0, 2)
        r = self.relation_embeds(nf3_data[:, 1])
        return self.inclusion_loss(c_boxes.translate(r), d_boxes)

    def nf3_neg_loss(self, neg_data):
        c_boxes, d_boxes = self.get_class_boxes(neg_data, 0, 2)
        r = self.relation_embeds(neg_data[:, 1])
        return self.neg_loss(c_boxes.translate(r), d_boxes)

    def nf4_loss(self, nf4_data):
        c_boxes, d_boxes = self.get_class_boxes(nf4_data, 1, 2)
        r = self.relation_embeds(nf4_data[:, 0])
        return self.inclusion_loss(c_boxes.translate(-r), d_boxes)

    def get_data_batch(self, train_data, key):
        if len(train_data[key]) <= self.batch_size:
            return train_data[key].to(self.device)
        else:
            rand_index = np.random.choice(len(train_data[key]), size=self.batch_size)
            return train_data[key][rand_index].to(self.device)

    def get_negative_sample_batch(self, train_data, key):
        rand_index = np.random.choice(len(train_data[f'{key}0']), size=self.batch_size)
        neg_data = train_data[f'{key}0'][rand_index]
        for i in range(1, self.num_neg):
            neg_data2 = train_data[f'{key}{i}'][rand_index]
            neg_data = torch.cat([neg_data, neg_data2], dim=0)
        return neg_data.to(self.device)

    def forward(self, train_data):
        loss = 0

        nf1_data = self.get_data_batch(train_data, 'nf1')
        loss += self.nf1_loss(nf1_data).square().mean()

        if len(train_data['nf2']) > 0:
            nf2_data = self.get_data_batch(train_data, 'nf2')
            loss += self.nf2_loss(nf2_data).square().mean()

        nf3_data = self.get_data_batch(train_data, 'nf3')
        loss += self.nf3_loss(nf3_data).square().mean()

        if len(train_data['nf4']) > 0:
            nf4_data = self.get_data_batch(train_data, 'nf4')
            loss += self.nf4_loss(nf4_data).square().mean()

        if len(train_data['disjoint']) > 0:
            disjoint_data = self.get_data_batch(train_data, 'disjoint')
            loss += self.nf2_disjoint_loss(disjoint_data).square().mean()

        if self.num_neg > 0:
            neg_data = self.get_negative_sample_batch(train_data, 'nf3_neg')
            neg_loss = self.nf3_neg_loss(neg_data)
            loss += (self.neg_dist - neg_loss).square().mean()
        return loss

    def to_loaded_model(self):
        model = ElbeLoadedModel()
        model.embedding_size = self.embedding_dim
        model.class_embeds = self.class_embeds.weight.detach()
        model.relation_embeds = self.relation_embeds.weight.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''
        np.save(f'{folder}/class_embeds{suffix}.npy', self.class_embeds.weight.detach().cpu().numpy())
        np.save(f'{folder}/relations{suffix}.npy', self.relation_embeds.weight.detach().cpu().numpy())
