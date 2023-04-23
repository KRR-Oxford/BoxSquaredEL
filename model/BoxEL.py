import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform
import numpy as np
from model.loaded_models import BoxELLoadedModel

eps = 1e-8


class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed


def l2_side_regularizer(box, log_scale: bool = True):
    """Applies l2 regularization on all sides of all boxes and returns the sum.
    """
    min_x = box.min_embed
    delta_x = box.delta_embed

    if not log_scale:
        return torch.mean(delta_x ** 2)
    else:
        return torch.mean(F.relu(min_x + delta_x - 1 + eps)) + F.relu(torch.norm(min_x, p=2) - 1)


class BoxEL(nn.Module):

    def __init__(self, device, class_, relationNum, embedding_dim):
        super(BoxEL, self).__init__()

        min_init_value = [1e-4, 0.2]
        delta_init_value = [-0.1, 0]
        relation_init_value = [-0.1, 0.1]
        scaling_init_value = [0.9, 1.1]
        vocab_size = len(class_)
        min_embedding = self.init_concept_embedding(vocab_size, embedding_dim, min_init_value)
        delta_embedding = self.init_concept_embedding(vocab_size, embedding_dim, delta_init_value)
        relation_embedding = self.init_concept_embedding(relationNum, embedding_dim, relation_init_value)
        scaling_embedding = self.init_concept_embedding(relationNum, embedding_dim, scaling_init_value)

        self.name = 'boxel'
        self.device = device
        self.embedding_dim = embedding_dim
        self.temperature = 1
        self.negative_sampling = False
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)
        self.relation_embedding = nn.Parameter(relation_embedding)
        self.scaling_embedding = nn.Parameter(scaling_embedding)

    def init_concept_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed

    def forward(self, input):
        batch = 512

        rand_index = np.random.choice(len(input['nf1']), size=batch)
        nf1_data = input['nf1'][rand_index]
        nf1_data = nf1_data.to(self.device)
        nf1_min = self.min_embedding[nf1_data]
        nf1_delta = self.delta_embedding[nf1_data]
        nf1_max = nf1_min + torch.exp(nf1_delta)
        boxes1 = Box(nf1_min[:, 0, :], nf1_max[:, 0, :])
        boxes2 = Box(nf1_min[:, 1, :], nf1_max[:, 1, :])
        nf1_loss, nf1_reg_loss = self.nf1_loss(boxes1, boxes2)

        if len(input['nf2']) == 0:
            nf2_loss = nf2_reg_loss = torch.as_tensor(0)
        else:
            rand_index = np.random.choice(len(input['nf2']), size=batch)
            nf2_data = input['nf2'][rand_index]
            nf2_data = nf2_data.to(self.device)
            nf2_min = self.min_embedding[nf2_data]
            nf2_delta = self.delta_embedding[nf2_data]
            nf2_max = nf2_min + torch.exp(nf2_delta)
            boxes1 = Box(nf2_min[:, 0, :], nf2_max[:, 0, :])
            boxes2 = Box(nf2_min[:, 1, :], nf2_max[:, 1, :])
            boxes3 = Box(nf2_min[:, 2, :], nf2_max[:, 2, :])
            nf2_loss, nf2_reg_loss = self.nf2_loss(boxes1, boxes2, boxes3)

        rand_index = np.random.choice(len(input['nf3']), size=batch)
        nf3_data = input['nf3'][rand_index]
        nf3_data = nf3_data.to(self.device)
        nf3_min = self.min_embedding[nf3_data[:, [0, 2]]]
        nf3_delta = self.delta_embedding[nf3_data[:, [0, 2]]]
        nf3_max = nf3_min + torch.exp(nf3_delta)
        relation = self.relation_embedding[nf3_data[:, 1]]
        scaling = self.scaling_embedding[nf3_data[:, 1]]
        boxes1 = Box(nf3_min[:, 0, :], nf3_max[:, 0, :])
        boxes2 = Box(nf3_min[:, 1, :], nf3_max[:, 1, :])
        nf3_loss, nf3_reg_loss = self.nf3_loss(boxes1, relation, scaling, boxes2)

        if len(input['nf4']) == 0:
            nf4_loss = nf4_reg_loss = torch.as_tensor(0)
        else:
            rand_index = np.random.choice(len(input['nf4']), size=batch)
            nf4_data = input['nf4'][rand_index]
            nf4_data = nf4_data.to(self.device)
            nf4_min = self.min_embedding[nf4_data[:, 1:]]
            nf4_delta = self.delta_embedding[nf4_data[:, 1:]]
            nf4_max = nf4_min + torch.exp(nf4_delta)
            relation = self.relation_embedding[nf4_data[:, 0]]
            scaling = self.scaling_embedding[nf4_data[:, 0]]
            boxes1 = Box(nf4_min[:, 0, :], nf4_max[:, 0, :])
            boxes2 = Box(nf4_min[:, 1, :], nf4_max[:, 1, :])
            nf4_loss, nf4_reg_loss = self.nf4_loss(relation, scaling, boxes1, boxes2)

        if len(input['disjoint']) == 0:
            disjoint_loss = disjoint_reg_loss = torch.as_tensor(0)
        else:
            rand_index = np.random.choice(len(input['disjoint']), size=batch)
            disjoint_data = input['disjoint'][rand_index]
            disjoint_data = disjoint_data.to(self.device)
            disjoint_min = self.min_embedding[disjoint_data]
            disjoint_delta = self.delta_embedding[disjoint_data]
            disjoint_max = disjoint_min + torch.exp(disjoint_delta)
            boxes1 = Box(disjoint_min[:, 0, :], disjoint_max[:, 0, :])
            boxes2 = Box(disjoint_min[:, 1, :], disjoint_max[:, 1, :])
            disjoint_loss, disjoint_reg_loss = self.disjoint_loss(boxes1, boxes2)

        if len(input['nf3_neg0']) == 0:
            neg_loss = neg_reg_loss = torch.as_tensor(0)
        else:
            rand_index = np.random.choice(len(input['nf3_neg0']), size=batch)
            neg_data = input['nf3_neg0'][rand_index]
            neg_data = neg_data.to(self.device)
            nf3_neg_min = self.min_embedding[neg_data[:, [0, 2]]]
            nf3_neg_delta = self.delta_embedding[neg_data[:, [0, 2]]]
            nf3_neg_max = nf3_neg_min + torch.exp(nf3_neg_delta)
            relation = self.relation_embedding[neg_data[:, 1]]
            scaling = self.scaling_embedding[neg_data[:, 1]]
            boxes1 = Box(nf3_neg_min[:, 0, :], nf3_neg_max[:, 0, :])
            boxes2 = Box(nf3_neg_min[:, 1, :], nf3_neg_max[:, 1, :])
            neg_loss, neg_reg_loss = self.nf3_neg_loss(boxes1, relation, scaling, boxes2)

        total_loss = [nf1_loss.sum() + nf2_loss.sum() + nf3_loss.sum() + nf4_loss.sum() + disjoint_loss.sum() + \
                     neg_loss.sum() + nf1_reg_loss + nf2_reg_loss + nf3_reg_loss + nf4_reg_loss + disjoint_reg_loss \
                     + neg_reg_loss]
        return total_loss

    @staticmethod
    def volumes(boxes):
        return F.softplus(boxes.delta_embed, beta=1).prod(1)

    @staticmethod
    def intersection(boxes1, boxes2):
        intersections_min = torch.max(boxes1.min_embed, boxes2.min_embed)
        intersections_max = torch.min(boxes1.max_embed, boxes2.max_embed)
        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    def inclusion_loss(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_box1 = torch.log(torch.clamp(self.volumes(boxes1), 1e-10, 1e4))

        return 1 - torch.exp(log_intersection - log_box1)

    def nf1_loss(self, boxes1, boxes2):
        return self.inclusion_loss(boxes1, boxes2), l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(
            boxes2, log_scale=True)

    def nf2_loss(self, boxes1, boxes2, boxes3):
        inter_box = self.intersection(boxes1, boxes2)
        return self.inclusion_loss(inter_box, boxes3), l2_side_regularizer(inter_box,
                                                                           log_scale=True) + l2_side_regularizer(boxes1,
                                                                                                                 log_scale=True) + l2_side_regularizer(
            boxes2, log_scale=True) + l2_side_regularizer(boxes3, log_scale=True)

    def nf3_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed * (scaling + eps) + relation
        trans_max = boxes1.max_embed * (scaling + eps) + relation
        trans_boxes = Box(trans_min, trans_max)
        return self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes,
                                                                             log_scale=True) + l2_side_regularizer(
            boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)

    def nf4_loss(self, relation, scaling, boxes1, boxes2):
        trans_min = (boxes1.min_embed - relation) / (scaling + eps)
        trans_max = (boxes1.max_embed - relation) / (scaling + eps)
        trans_boxes = Box(trans_min, trans_max)
        #         log_trans_boxes = torch.log(torch.clamp(self.volumes(trans_boxes), 1e-10, 1e4))
        return self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes,
                                                                             log_scale=True) + l2_side_regularizer(
            boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)

    def disjoint_loss(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_boxes1 = torch.log(torch.clamp(self.volumes(boxes1), 1e-10, 1e4))
        log_boxes2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        union = log_boxes1 + log_boxes2
        return torch.exp(log_intersection - union), l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(
            boxes2, log_scale=True)

    def nf3_neg_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed * (scaling + eps) + relation
        trans_max = boxes1.max_embed * (scaling + eps) + relation
        trans_boxes = Box(trans_min, trans_max)
        #         trans_min = boxes1.min_embed + relation
        #         trans_max = trans_min + torch.clamp((boxes1.max_embed - boxes1.min_embed)*(scaling + eps), 1e-10, 1e4)
        #         trans_boxes = Box(trans_min, trans_max)
        return 1 - self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes,
                                                                                 log_scale=True) + l2_side_regularizer(
            boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)

    def to_loaded_model(self):
        model = BoxELLoadedModel()
        model.embedding_size = self.embedding_dim
        model.min_embedding = self.min_embedding.detach()
        model.delta_embedding = self.delta_embedding.detach()
        model.relation_embedding = self.relation_embedding.detach()
        model.scaling_embedding = self.scaling_embedding.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''
        np.save(f'{folder}/min_embeds{suffix}.npy', self.min_embedding.detach().cpu().numpy())
        np.save(f'{folder}/delta_embeds{suffix}.npy', self.delta_embedding.detach().cpu().numpy())
        np.save(f'{folder}/rel_embeds{suffix}.npy', self.relation_embedding.detach().cpu().numpy())
        np.save(f'{folder}/scaling_embeds{suffix}.npy', self.scaling_embedding.detach().cpu().numpy())
