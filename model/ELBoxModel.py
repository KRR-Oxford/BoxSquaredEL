import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu

np.random.seed(12)


class ELBoxModel(nn.Module):

    def __init__(self, device, class_, relationNum, embedding_dim, batch, margin=0, disjoint_dist=2,
                 ranking_fn='l2'):
        super(ELBoxModel, self).__init__()

        self.margin = margin
        self.disjoint_dist = disjoint_dist
        self.classNum = len(class_)
        self.class_ = class_
        self.relationNum = relationNum
        self.device = device
        self.beta = None
        self.ranking_fn = ranking_fn

        self.classEmbeddingDict = nn.Embedding(self.classNum, embedding_dim * 2)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight.data, axis=1).reshape(
            -1, 1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        nn.init.uniform_(self.relationEmbeddingDict.weight, a=-1, b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight.data, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim

    # cClass isSubSetof dClass

    def nf1Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        cr = torch.abs(c[:, self.embedding_dim:])
        dr = torch.abs(d[:, self.embedding_dim:])

        euc = torch.abs(c1 - d1)

        dst = torch.reshape(torch.linalg.norm(relu(euc + cr - dr - self.margin), axis=1), [-1, 1])

        return dst

    # cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])
        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]
        e1 = e[:, :self.embedding_dim]

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 = torch.abs(d[:, self.embedding_dim:])
        e2 = torch.abs(e[:, self.embedding_dim:])

        startAll = torch.maximum(c1 - c2, d1 - d2)
        endAll = torch.minimum(c1 + c2, d1 + d2)

        newR = torch.abs(startAll - endAll) / 2

        cen1 = (startAll + endAll) / 2
        cen2 = e1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc + newR - e2 - self.margin), axis=1), [-1, 1]) \
              + torch.linalg.norm(relu(startAll - endAll), axis=1)

        return dst

    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        cr = torch.abs(c[:, self.embedding_dim:])
        dr = torch.abs(d[:, self.embedding_dim:])

        cen1 = c1
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc - cr - dr + self.margin), axis=1), [-1, 1])

        return dst

    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c_center = c[:, :self.embedding_dim]
        c_offset = torch.abs(c[:, self.embedding_dim:])

        d_center = d[:, :self.embedding_dim]
        d_offset = torch.abs(d[:, self.embedding_dim:])

        cen1 = c_center + r
        cen2 = d_center
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc + c_offset - d_offset - self.margin), axis=1), [-1, 1])

        return dst

    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c_center = c[:, :self.embedding_dim]
        c_offset = torch.abs(c[:, self.embedding_dim:])

        d_center = d[:, :self.embedding_dim]
        d_offset = torch.abs(d[:, self.embedding_dim:])

        cen1 = c_center + r
        cen2 = d_center
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc - c_offset - d_offset + self.margin), axis=1), [-1, 1])

        return dst

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])
        r = self.relationEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 2])

        c_center = c[:, :self.embedding_dim]
        c_offset = torch.abs(c[:, self.embedding_dim:])

        d_center = d[:, :self.embedding_dim]
        d_offset = torch.abs(d[:, self.embedding_dim:])

        cen1 = c_center - r
        cen2 = d_center
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc - c_offset - d_offset - self.margin), axis=1), [-1, 1])

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
        negLoss = (self.disjoint_dist - self.neg_loss(negData)).square().mean()

        totalLoss = [loss1 + loss2 + disJointLoss + loss3 + loss4 + negLoss]
        return totalLoss
