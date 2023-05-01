import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu
from boxes import Boxes
import os
from model.loaded_models import BoxSqELLoadedModel


class BoxSquaredEL(nn.Module):
    def __init__(self, device, embedding_dim, num_classes, num_roles, num_individuals=0, margin=0, neg_dist=2,
                 reg_factor=0.05, num_neg=2, batch_size=512, vis_loss=False):
        super(BoxSquaredEL, self).__init__()

        self.name = 'boxsqel'
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_roles = num_roles
        self.num_individuals = num_individuals
        self.margin = margin
        self.neg_dist = neg_dist
        self.reg_factor = reg_factor
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.vis_loss = vis_loss

        self.negative_sampling = True

        self.class_embeds = self.init_embeddings(self.num_classes, embedding_dim * 2)
        self.individual_embeds = self.init_embeddings(self.num_individuals, embedding_dim)
        self.bumps = self.init_embeddings(self.num_classes, embedding_dim)
        self.individual_bumps = self.init_embeddings(self.num_individuals, embedding_dim)
        self.relation_heads = self.init_embeddings(num_roles, embedding_dim * 2)
        self.relation_tails = self.init_embeddings(num_roles, embedding_dim * 2)

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

    def get_relation_boxes(self, nf_data, *indices):
        boxes = []
        for i in indices:
            boxes.append(self.get_boxes(self.relation_heads(nf_data[:, i])))
            boxes.append(self.get_boxes(self.relation_tails(nf_data[:, i])))
        return tuple(boxes)

    def get_individual_boxes(self, nf_data, *indices):
        """Returns a representation of individuals as boxes with an offset/volume of 0."""
        return (
            Boxes(self.individual_embeds(nf_data[:, i]),
                  torch.zeros((nf_data.shape[0], self.embedding_dim)).to(self.device))
            for i in indices
        )

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
        c_bumps, d_bumps = self.bumps(nf3_data[:, 0]), self.bumps(nf3_data[:, 2])
        head_boxes, tail_boxes = self.get_relation_boxes(nf3_data, 1)

        dist1 = self.inclusion_loss(c_boxes.translate(d_bumps), head_boxes)
        dist2 = self.inclusion_loss(d_boxes.translate(c_bumps), tail_boxes)
        return (dist1 + dist2) / 2

    def nf3_neg_loss(self, neg_data):
        c_boxes, d_boxes = self.get_class_boxes(neg_data, 0, 2)
        c_bumps, d_bumps = self.bumps(neg_data[:, 0]), self.bumps(neg_data[:, 2])
        head_boxes, tail_boxes = self.get_relation_boxes(neg_data, 1)

        return self.neg_loss(c_boxes.translate(d_bumps), head_boxes), \
               self.neg_loss(d_boxes.translate(c_bumps), tail_boxes)

    def role_assertion_loss(self, data):
        a_boxes, b_boxes = self.get_individual_boxes(data, 1, 2)
        a_bumps, b_bumps = self.individual_bumps(data[:, 1]), self.individual_bumps(data[:, 2])
        head_boxes, tail_boxes = self.get_relation_boxes(data, 0)

        dist1 = self.inclusion_loss(a_boxes.translate(b_bumps), head_boxes)
        dist2 = self.inclusion_loss(b_boxes.translate(a_bumps), tail_boxes)
        return (dist1 + dist2) / 2

    def role_assertion_neg_loss(self, neg_data):
        a_boxes, b_boxes = self.get_individual_boxes(neg_data, 1, 2)
        a_bumps, b_bumps = self.individual_bumps(neg_data[:, 1]), self.individual_bumps(neg_data[:, 2])
        head_boxes, tail_boxes = self.get_relation_boxes(neg_data, 0)

        return self.neg_loss(a_boxes.translate(b_bumps), head_boxes), \
               self.neg_loss(b_boxes.translate(a_bumps), tail_boxes)

    def concept_assertion_loss(self, data):
        a_boxes, = self.get_individual_boxes(data, 2)
        a_bumps = self.individual_bumps(data[:, 2])
        c_boxes, = self.get_class_boxes(data, 1)
        c_bumps = self.bumps(data[:, 1])
        head_boxes, tail_boxes = self.get_relation_boxes(data, 0)

        dist1 = self.inclusion_loss(a_boxes.translate(c_bumps), head_boxes)
        dist2 = self.inclusion_loss(c_boxes.translate(a_bumps), tail_boxes)
        return (dist1 + dist2) / 2

    def nf4_loss(self, nf4_data):
        d_boxes, = self.get_class_boxes(nf4_data, 2)
        c_bumps = self.bumps(nf4_data[:, 1])
        head_boxes, _ = self.get_relation_boxes(nf4_data, 0)

        return self.inclusion_loss(head_boxes.translate(-c_bumps), d_boxes)

    def get_nf_data_batch(self, train_data, nf_key):
        rand_index = np.random.choice(len(train_data[nf_key]), size=self.batch_size)
        return train_data[nf_key][rand_index].to(self.device)

    def get_negative_sample_batch(self, train_data, key):
        rand_index = np.random.choice(len(train_data[f'{key}0']), size=self.batch_size)
        neg_data = train_data[f'{key}0'][rand_index]
        for i in range(1, self.num_neg):
            neg_data2 = train_data[f'{key}{i}'][rand_index]
            neg_data = torch.cat([neg_data, neg_data2], dim=0)
        return neg_data.to(self.device)

    def forward(self, train_data):
        loss = 0

        nf1_data = self.get_nf_data_batch(train_data, 'nf1')
        loss += self.nf1_loss(nf1_data).square().mean()

        if len(train_data['nf2']) > 0:
            nf2_data = self.get_nf_data_batch(train_data, 'nf2')
            loss += self.nf2_loss(nf2_data).square().mean()

        nf3_data = self.get_nf_data_batch(train_data, 'nf3')
        loss += self.nf3_loss(nf3_data).square().mean()

        if len(train_data['nf4']) > 0:
            nf4_data = self.get_nf_data_batch(train_data, 'nf4')
            loss += self.nf4_loss(nf4_data).square().mean()

        if len(train_data['disjoint']) > 0:
            disjoint_data = self.get_nf_data_batch(train_data, 'disjoint')
            loss += self.nf2_disjoint_loss(disjoint_data).square().mean()

        if self.num_neg > 0:
            neg_data = self.get_negative_sample_batch(train_data, 'nf3_neg')
            neg_loss1, neg_loss2 = self.nf3_neg_loss(neg_data)
            loss += (self.neg_dist - neg_loss1).square().mean() + (self.neg_dist - neg_loss2).square().mean()

        if 'abox' in train_data:
            abox = train_data['abox']
            ra_data = self.get_nf_data_batch(abox, 'role_assertions')
            loss += self.role_assertion_loss(ra_data).square().mean()

            neg_data = self.get_negative_sample_batch(abox, 'role_assertions_neg')
            neg_loss1, neg_loss2 = self.role_assertion_neg_loss(neg_data)
            loss += (self.neg_dist - neg_loss1).square().mean() + (self.neg_dist - neg_loss2).square().mean()

            ca_data = self.get_nf_data_batch(abox, 'concept_assertions')
            loss += self.concept_assertion_loss(ca_data).square().mean()

        class_reg = self.reg_factor * torch.linalg.norm(self.bumps.weight, dim=1).reshape(-1, 1).mean()
        if self.num_individuals > 0:
            individual_reg = \
                self.reg_factor * torch.linalg.norm(self.individual_bumps.weight, dim=1).reshape(-1, 1).mean()
            loss += (class_reg + individual_reg) / 2
        else:
            loss += class_reg

        if self.vis_loss:  # only used for plotting nice boxes
            vis_loss = relu(.2 - torch.abs(self.class_embeds.weight[:, self.embedding_dim:]))
            loss += vis_loss.mean()

        return loss

    def to_loaded_model(self):
        model = BoxSqELLoadedModel()
        model.embedding_size = self.embedding_dim
        model.class_embeds = self.class_embeds.weight.detach()
        model.bumps = self.bumps.weight.detach()
        model.relation_heads = self.relation_heads.weight.detach()
        model.relation_tails = self.relation_tails.weight.detach()
        if self.num_individuals > 0:
            model.individual_embeds = self.individual_embeds.weight.detach()
            model.individual_bumps = self.individual_bumps.weight.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''
        np.save(f'{folder}/class_embeds{suffix}.npy', self.class_embeds.weight.detach().cpu().numpy())
        np.save(f'{folder}/bumps{suffix}.npy', self.bumps.weight.detach().cpu().numpy())
        np.save(f'{folder}/rel_heads{suffix}.npy', self.relation_heads.weight.detach().cpu().numpy())
        np.save(f'{folder}/rel_tails{suffix}.npy', self.relation_tails.weight.detach().cpu().numpy())
        if self.num_individuals > 0:
            np.save(f'{folder}/individual_embeds{suffix}.npy', self.individual_embeds.weight.detach().cpu().numpy())
            np.save(f'{folder}/individual_bumps{suffix}.npy', self.individual_bumps.weight.detach().cpu().numpy())
