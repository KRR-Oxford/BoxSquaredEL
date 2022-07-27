import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu
from boxes import Boxes

np.random.seed(12)


class Box2ELModel(nn.Module):

    def __init__(self, device, class_, relationNum, embedding_dim, batch, margin=0, disjoint_dist=2,
                 ranking_fn='l2'):
        super(Box2ELModel, self).__init__()

        self.margin = margin
        self.disjoint_dist = disjoint_dist
        self.classNum = len(class_)
        self.class_ = class_
        self.relationNum = relationNum
        self.device = device
        self.beta = None
        self.ranking_fn = ranking_fn
        self.embedding_dim = embedding_dim

        self.classEmbeddingDict = self.init_embeddings(self.classNum, embedding_dim * 2)
        self.bumps = self.init_embeddings(self.classNum, embedding_dim)
        self.relation_heads = self.init_embeddings(relationNum, embedding_dim * 2)
        self.relation_tails = self.init_embeddings(relationNum, embedding_dim * 2)

    def init_embeddings(self, num_embeddings, dim, min=-1, max=1, normalise=True):
        embeddings = nn.Embedding(num_embeddings, dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        if normalise:
            embeddings.weight.data /= torch.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        return embeddings

    def get_boxes(self, embedding):
        return Boxes(embedding[:, :self.embedding_dim], torch.abs(embedding[:, self.embedding_dim:]))

    def nf1Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)

        euc = torch.abs(c_boxes.centers - d_boxes.centers)
        dst = torch.reshape(torch.linalg.norm(relu(euc + c_boxes.offsets - d_boxes.offsets - self.margin), axis=1),
                            [-1, 1])
        return dst

    def nf2Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)
        e_boxes = self.get_boxes(e)

        intersection_lower_left = torch.maximum(c_boxes.centers - c_boxes.offsets, d_boxes.centers - d_boxes.offsets)
        intersection_upper_right = torch.minimum(c_boxes.centers + c_boxes.offsets, d_boxes.centers + d_boxes.offsets)
        intersection_offsets = torch.abs(intersection_lower_left - intersection_upper_right) / 2
        intersection_center = (intersection_lower_left + intersection_upper_right) / 2

        euc = torch.abs(intersection_center - e_boxes.centers)
        dst = torch.reshape(
            torch.linalg.norm(relu(euc + intersection_offsets - e_boxes.offsets - self.margin), axis=1), [-1, 1])
        return dst + torch.linalg.norm(relu(intersection_lower_left - intersection_upper_right), axis=1)

    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)

        euc = torch.abs(c_boxes.centers - d_boxes.centers)
        dst = torch.reshape(torch.linalg.norm(relu(euc - c_boxes.offsets - d_boxes.offsets + self.margin), axis=1),
                            [-1, 1])
        return dst

    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 2])
        c_bumps = self.bumps(input[:, 0])
        d_bumps = self.bumps(input[:, 2])
        r_heads = self.relation_heads(input[:, 1])
        r_tails = self.relation_tails(input[:, 1])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)
        head_boxes = self.get_boxes(r_heads)
        tail_boxes = self.get_boxes(r_tails)

        euc1 = torch.abs(c_boxes.centers + d_bumps - head_boxes.centers)
        dst1 = torch.reshape(torch.linalg.norm(relu(euc1 + c_boxes.offsets - head_boxes.offsets - self.margin), axis=1),
                             [-1, 1])
        euc2 = torch.abs(d_boxes.centers + c_bumps - tail_boxes.centers)
        dst2 = torch.reshape(torch.linalg.norm(relu(euc2 + d_boxes.offsets - tail_boxes.offsets - self.margin), axis=1),
                             [-1, 1])
        return (dst1 + dst2) / 2

    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 2])
        c_bumps = self.bumps(input[:, 0])
        d_bumps = self.bumps(input[:, 2])
        r_heads = self.relation_heads(input[:, 1])
        r_tails = self.relation_tails(input[:, 1])

        c_boxes = self.get_boxes(c)
        d_boxes = self.get_boxes(d)
        head_boxes = self.get_boxes(r_heads)
        tail_boxes = self.get_boxes(r_tails)

        euc1 = torch.abs(c_boxes.centers + d_bumps - head_boxes.centers)
        dst1 = torch.reshape(torch.linalg.norm(relu(euc1 - c_boxes.offsets - head_boxes.offsets + self.margin), axis=1),
                             [-1, 1])
        euc2 = torch.abs(d_boxes.centers + c_bumps - tail_boxes.centers)
        dst2 = torch.reshape(torch.linalg.norm(relu(euc2 - d_boxes.offsets - tail_boxes.offsets + self.margin), axis=1),
                             [-1, 1])
        return dst1, dst2

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        d = self.classEmbeddingDict(input[:, 2])
        c_bumps = self.bumps(input[:, 1])
        r_heads = self.relation_heads(input[:, 0])

        d_boxes = self.get_boxes(d)
        head_boxes = self.get_boxes(r_heads)

        euc = torch.abs(head_boxes.centers - c_bumps - d_boxes.centers)
        dst = torch.reshape(torch.linalg.norm(relu(euc + head_boxes.offsets - d_boxes.offsets - self.margin), axis=1),
                            [-1, 1])
        return dst

    def forward(self, input):
        batch = 512

        rand_index = np.random.choice(len(input['nf1']), size=batch)
        nf1Data = input['nf1'][rand_index]
        nf1Data = nf1Data.to(self.device)
        loss1 = self.nf1Loss(nf1Data).square().mean()

        # nf2
        rand_index = np.random.choice(len(input['nf2']), size=batch)
        nf2Data = input['nf2'][rand_index]
        nf2Data = nf2Data.to(self.device)
        loss2 = self.nf2Loss(nf2Data).square().mean()

        # nf3
        rand_index = np.random.choice(len(input['nf3']), size=batch)
        nf3Data = input['nf3'][rand_index]
        nf3Data = nf3Data.to(self.device)
        loss3 = self.nf3Loss(nf3Data).square().mean()

        # nf4
        rand_index = np.random.choice(len(input['nf4']), size=batch)
        nf4Data = input['nf4'][rand_index]
        nf4Data = nf4Data.to(self.device)
        loss4 = self.nf4Loss(nf4Data).square().mean()

        # disJoint
        if len(input['disjoint']) == 0:
            disJointLoss = 0
        else:
            rand_index = np.random.choice(len(input['disjoint']), size=batch)
            disJointData = input['disjoint'][rand_index]
            disJointData = disJointData.to(self.device)
            disJointLoss = (self.disjoint_dist - self.disJointLoss(disJointData)).relu().square().mean()

        # negLoss
        rand_index = np.random.choice(len(input['nf3_neg']), size=batch)
        negData = input['nf3_neg'][rand_index]
        negData = negData.to(self.device)
        negLoss1, negLoss2 = self.neg_loss(negData)
        negLoss = (self.disjoint_dist - negLoss1).square().mean() + (self.disjoint_dist - negLoss2).square().mean()
        # negLoss /= 2

        totalLoss = [loss1 + loss2 + disJointLoss + loss3 + loss4 + negLoss]
        return totalLoss
