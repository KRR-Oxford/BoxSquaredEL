import random

import torch.nn as nn
import torch
from model.TransR import TransR
import numpy as np
import random
class RelationModel(nn.Module):
    def __init__(self, embedding_dim):
        super(RelationModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(  embedding_dim, 16 * embedding_dim),
            nn.Tanh(),
            nn.Linear(16 * embedding_dim, 32  * embedding_dim),
            nn.Tanh(),
            nn.Linear(32 * embedding_dim, 16 * embedding_dim),
            nn.Tanh(),
            nn.Linear(16 * embedding_dim, embedding_dim),
            nn.Tanh(),
        )
    def forward(self,input):
        return self.mlp(input)
class ELBox2BallModel(nn.Module):
    '''

    Args:
        classNum: number of classes
        relationNum: number of relations
        embedding_dim: the dimension of the embedding(both class and relatio)
        margin: the distance that two box apart
    '''

    def __init__(self, device, classNum, relationNum, embedding_dim, margin=0):
        super(ELBox2BallModel, self).__init__()
        self.margin = margin
        self.classNum = classNum
        self.TransR = TransR(embedding_dim)
        self.relationNum = relationNum
        self.inf = 5
        self.device = device

        self.transfer_matrix = nn.Embedding(relationNum, embedding_dim *embedding_dim)
        nn.init.uniform_(self.transfer_matrix.weight, a=-1, b=1)
        self.transfer_matrix.weight.data /= torch.linalg.norm(
            self.transfer_matrix.weight, axis=1).reshape(-1, 1)

        self.classEmbeddingDict = nn.Embedding(classNum, embedding_dim *2)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight, axis=1).reshape(-1, 1)
       # self.classEmbeddingDict.weight.data*= 2

        self.reg_norm = 1

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        nn.init.uniform_(self.relationEmbeddingDict.weight, a=-1, b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim
        self.relationModel = RelationModel(embedding_dim)

    def reg(self, x):
        res = torch.abs(torch.linalg.norm(x, axis=1) - self.reg_norm)
        res = torch.reshape(res, [-1, 1])
        return res

    # cClass isSubSetof dClass
    def nf1Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])


        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = c[:, self.embedding_dim:]
        d2 = d[:, self.embedding_dim:]

        # box

        margin = (torch.ones(c1.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(c1.shape, requires_grad=False) * self.margin).to(self.device)
        relu1 = nn.ReLU()
        leftBottomLimit = torch.reshape(torch.linalg.norm(relu1(d1 - c1 + margin), axis=1), [-1, 1])
        relu2 = nn.ReLU()
        righttopLimit = torch.reshape(torch.linalg.norm(relu2(c1+c2 - d2-d1 + margin), axis=1), [-1, 1])

        relu3 = nn.ReLU()
        shapeLimit = torch.reshape(torch.linalg.norm(relu3( - c2
                                                     ), axis=1,ord=1), [-1, 1])
        relu4 = nn.ReLU()
        shapeLimit += torch.reshape(torch.linalg.norm(relu4( - d2
                                                      ), axis=1, ord = 1), [-1, 1])
        return leftBottomLimit+righttopLimit+ shapeLimit #+ self.reg(c1) + self.reg(c2+c1) + self.reg(d1) + self.reg(d2+d1)

        # return  dst + self.reg(x1) + self.reg(x2)+ shapeLimit

    # cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])
        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]
        e1 = e[:, :self.embedding_dim]

        c2 = c[:, self.embedding_dim:]
        d2 = d[:, self.embedding_dim:]
        e2 = e[:, self.embedding_dim:]

        relu1 = torch.nn.ReLU()
        relu2 = torch.nn.ReLU()
        relu3 = torch.nn.ReLU()
        relu4 = torch.nn.ReLU()
        relu5 = torch.nn.ReLU()

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c1+c2, d1+d2)
       # endAll = torch.maximum(endAll-startAll,torch.zeros(startAll.shape).to(self.device))+startAll

        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)
        # print(margin.shape)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.reshape(torch.linalg.norm(relu1(e1 - startAll ), axis=1),[-1,1])
        righttopLimit = torch.reshape(torch.linalg.norm(relu2(endAll - e2-e1 ), axis=1),[-1,1])

        shapeLimit = torch.reshape(torch.linalg.norm(relu3(- c2 + margin
                                                           ), axis=1),[-1,1])

        shapeLimit += torch.reshape(torch.linalg.norm(relu4( - d2 + margin
                                                            ), axis=1),[-1,1])
        shapeLimit += torch.reshape(torch.linalg.norm(relu5( - e2 + margin
                                                            ), axis=1),[-1,1])


        return leftBottomLimit+righttopLimit+shapeLimit#+ self.reg(c1) + self.reg(c2+c1) + self.reg(d1)+ self.reg(d2+d1) + self.reg(e1)+ self.reg(e2+e1)
        # return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3) +shapeLimit



    # cClass isSubSet of relation some dClass
    # def nf3Loss(self, input):
    #     c = self.classEmbeddingDict(input[:, 0])
    #     r = self.relationEmbeddingDict(input[:, 1])
    #     d = self.classEmbeddingDict(input[:, 2])
    #     #
    #     c1 = c[:, :self.embedding_dim]  # .detach()
    #     c2 = c[:, self.embedding_dim:]  # .detach()
    #
    #     d1 = d[:, :self.embedding_dim]  # .detach()
    #     d2 = d[:, self.embedding_dim:]  # .detach()
    #
    #     # c1_o = c[:, :self.embedding_dim]  # .detach()
    #     # c2_o = c[:, self.embedding_dim:]  # .detach()
    #     #
    #     # d1_o = d[:, :self.embedding_dim]  # .detach()
    #     # d2_o = d[:, self.embedding_dim:]  # .detach()
    #     #
    #     # c1_input = torch.cat((c1_o, r), dim=1)
    #     # c2_input = torch.cat((c2_o, r), dim=1)
    #     #
    #     # d1_input = torch.cat((d1_o, r), dim=1)
    #     # d2_input = torch.cat((d2_o, r), dim=1)
    #     #
    #     # c1 = self.relationModel(c1_input)
    #     # c2 = self.relationModel(c2_input)
    #     #
    #     # d1 = self.relationModel(d1_input)
    #     # d2 = self.relationModel(d2_input)
    #     #
    #
    #
    #     # is subset
    #
    #     margin = (torch.ones(d1.shape, requires_grad=False) * self.margin).to(self.device)
    #     zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)
    #
    #
    #
    #     leftBottomLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - c1 -r  + margin, zeros), axis=1),[-1,1])
    #     righttopLimit =  torch.reshape(torch.linalg.norm(torch.maximum(c2+c1 +r - d2-d1  + margin, zeros), axis=1),[-1,1])
    #     shapeLimit =  torch.reshape(torch.linalg.norm(torch.maximum( - d2 + margin, zeros), axis=1),[-1,1])
    #
    #     shapeLimit +=  torch.reshape(torch.linalg.norm(torch.maximum(  - c2 + margin, zeros), axis=1),[-1,1])
    #
    #     return leftBottomLimit + righttopLimit + shapeLimit #+ self.reg(c1) + self.reg(c2) + self.reg(d1) + self.reg(d2)
    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        x1 = c[:, 0:self.embedding_dim]
        x2 = d[:, 0:self.embedding_dim]

        rc = torch.abs(c[:, -1])
        rd = torch.abs(d[:, -1])

        x3 = x1 + r
        euc = torch.linalg.norm(x3 - x2, axis=1)
        relu = torch.nn.ReLU()
        dst = torch.reshape(relu(euc + rc - rd + self.margin), [-1, 1])

        return dst + self.reg(x1) + self.reg(x2)

    # def  neg_loss(self, input):
    #     c = self.classEmbeddingDict(input[:, 0])
    #     r = self.relationEmbeddingDict(input[:, 1])
    #     d = self.classEmbeddingDict(input[:, 2])
    #
    #     c1 = c[:, :self.embedding_dim]
    #     d1 = d[:, :self.embedding_dim]
    #
    #     c2 = c[:, self.embedding_dim:]
    #     d2 = d[:, self.embedding_dim:]
    #
    #
    #     startAll = torch.maximum(c1+r, d1)
    #     endAll = torch.minimum(c2+r+c1, d2+d1)
    #     margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)
    #
    #     zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)
    #     relu = nn.ReLU()
    #
    #     rightLessLeftLoss = torch.reshape(torch.sum(endAll - startAll + margin, dim=1 ),
    #                                       [-1, 1])
    #
    #     shapeLoss = torch.reshape(torch.linalg.norm(torch.maximum(  - c2 + margin,
    #                                                               zeros), axis=1), [-1, 1])
    #
    #     shapeLoss += torch.reshape(torch.linalg.norm(torch.maximum( - d2 + margin,
    #                                                                zeros), axis=1), [-1, 1])
    #
    #     return rightLessLeftLoss
    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        x1 = c[:, 0:self.embedding_dim]
        x2 = d[:, 0:self.embedding_dim]
        rc = torch.abs(c[:, -1])
        rd = torch.abs(d[:, -1])

        x3 = x1 + r
        euc = torch.linalg.norm(x3 - x2, axis=1)

        #   relu = torch.nn
        dst = torch.reshape((-(euc - rc - rd) + self.margin), [-1, 1])
        return dst + self.reg(x1) + self.reg(x2)

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])

        r = self.relationEmbeddingDict(input[:, 0])

        d = self.classEmbeddingDict(input[:, 2])
        c1 = c[:, :self.embedding_dim]  # .detach()
        c2 = c[:, self.embedding_dim:]  # .detach()

        d1 = d[:, :self.embedding_dim]  # .detach()
        d2 = d[:, self.embedding_dim:]  # .detach()

        # c1_o = c[:, :self.embedding_dim]  # .detach()
        # c2_o = c[:, self.embedding_dim:]  # .detach()
        #
        # d1_o = d[:, :self.embedding_dim]  # .detach()
        # d2_o = d[:, self.embedding_dim:]  # .detach()
        #
        # c1_input = torch.cat((c1_o, r), dim=1)
        # c2_input = torch.cat((c2_o, r), dim=1)
        #
        # d1_input = torch.cat((d1_o, r), dim=1)
        # d2_input = torch.cat((d2_o, r), dim=1)
        #
        # c1 = self.relationModel(c1_input)
        # c2 = self.relationModel(c2_input)
        #
        # d1 = self.relationModel(d1_input)
        # d2 = self.relationModel(d2_input)

        # is subset
        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)

        leftBottomLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - c1+r + margin, zeros), axis=1),[-1,1])
        righttopLimit = torch.reshape(torch.linalg.norm(torch.maximum(c2+c1 -r - d2-d1 + margin, zeros), axis=1),[-1,1])

        shapeLimit = torch.reshape(torch.linalg.norm(torch.maximum(  - d2 + margin, zeros), axis=1),[-1,1])

        shapeLimit += torch.reshape(torch.linalg.norm(torch.maximum(  - c2 + margin, zeros), axis=1),[-1,1])

        return leftBottomLimit + righttopLimit + shapeLimit#+ self.reg(c1) + self.reg(c2) + self.reg(d1)+ self.reg(d2)
        # return dst + self.reg(x1) + self.reg(x2) + shapeLimit

    # def disJointLoss(self, input):
    #     c = self.classEmbeddingDict(input[:, 0])
    #     d = self.classEmbeddingDict(input[:, 1])
    #
    #     c1 = c[:, :self.embedding_dim]
    #     d1 = d[:, :self.embedding_dim]
    #
    #     c2 = c[:, self.embedding_dim:]
    #     d2 = d[:, self.embedding_dim:]
    #
    #     startAll = torch.maximum(c1, d1)
    #     endAll = torch.minimum(c2+c1, d2+d1)
    #     margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)
    #
    #
    #     relu1 = nn.ReLU()
    #     relu2 = nn.ReLU()
    #     relu3 = nn.ReLU()
    #
    #     rightLessLeftLoss = torch.reshape(torch.linalg.norm(relu1(endAll - startAll + margin), axis=1,ord = 1),[-1,1])
    #
    #     shapeLoss  = torch.reshape(torch.linalg.norm(relu2(  - c2 + margin
    #                                                 ), axis=1),[-1,1])
    #
    #     shapeLoss += torch.reshape(torch.linalg.norm(relu3(  - d2 + margin
    #                                                 ), axis=1),[-1,1])
    #
    #
    #     return rightLessLeftLoss + shapeLoss
    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        rc = torch.reshape(torch.abs(c[:, -1]), [-1, 1])
        rd = torch.reshape(torch.abs(d[:, -1]), [-1, 1])

        x1 = c[:, 0:self.embedding_dim]
        x2 = d[:, 0:self.embedding_dim]

        sr = rc + rd

        dst = torch.reshape(torch.linalg.norm(x2 - x1, axis=1), [-1, 1])
        relu = torch.nn.ReLU()
        return relu(sr - dst + self.margin) + self.reg(x1) + self.reg(x2)

    def top_loss(self, input):
        d = self.classEmbeddingDict(input)
       # print(input)
        d1 = d[:, :self.embedding_dim]

        d2 = d[:, self.embedding_dim:]
        margin = (torch.ones(d1.shape, requires_grad=False) * self.inf).to(self.device)

        zeros = (torch.zeros(d1.shape, requires_grad=False) * self.margin).to(self.device)
        shapeLoss = torch.reshape(torch.linalg.norm(torch.maximum(  - d2 + margin,
                                                                   zeros), axis=1), [-1, 1])
        shapeLoss += torch.reshape(torch.linalg.norm(torch.maximum(d1 + margin,
                                                                  zeros), axis=1), [-1, 1])
        return shapeLoss
    def forward(self, input):
        batch = 512
        # print(input['disjoint'])
        # nf1

        rand_index = np.random.choice(len(input['nf1']), size=batch)
        # print(len(input['nf1']))
        nf1Data = input['nf1'][rand_index]
        nf1Data = nf1Data.to(self.device)
        loss1 = self.nf1Loss(nf1Data)

        # nf2
        rand_index = np.random.choice(len(input['nf2']), size=batch)
        #   print(input['nf2'])
        nf2Data = input['nf2'][rand_index]
        nf2Data = nf2Data.to(self.device)
        loss2 = self.nf2Loss(nf2Data)

        # nf3
        rand_index = np.random.choice(len(input['nf3']), size=batch)

        nf3Data = input['nf3'][rand_index]
        nf3Data = nf3Data.to(self.device)
        loss3 = self.nf3Loss(nf3Data)

        # nf4
        rand_index = np.random.choice(len(input['nf4']), size=batch)
        nf4Data = input['nf4'][rand_index]
        nf4Data = nf4Data.to(self.device)
        loss4 = self.nf4Loss(nf4Data)

        # disJoint
        rand_index = np.random.choice(len(input['disjoint']), size=batch)

        disJointData = input['disjoint'][rand_index]
        disJointData = disJointData.to(self.device)
        disJointLoss = self.disJointLoss(disJointData)

        # top_loss
        rand_index = np.random.choice(len(input['top']), size=batch)
        topData = input['top'][rand_index]

        topData = topData.to(self.device)
        topLoss = self.top_loss(topData)

        # negLoss
        rand_index = np.random.choice(len(input['nf3_neg']), size=batch)
        negData = input['nf3_neg'][rand_index]
        negData = negData.to(self.device)
        negLoss = self.neg_loss(negData)

        # print(loss1,loss2, loss3, loss4, disJointLoss,
        #      negLoss)
        #  print( loss1,loss2,disJointLoss)

        totalLoss =negLoss+loss3+disJointLoss# loss1 + disJointLoss + loss2  +loss3+ loss4 +negLoss
        print('loss1=',(loss1).sum().item()/batch,'loss2=',(loss2).sum().item()/batch,'loss3=',(loss3).sum().item()/batch,
              'loss4=',(loss4).sum().item()/batch,'disJointLoss=',disJointLoss.sum().item()/batch,'negLoss=',negLoss.sum().item()/batch,)

        # print(torch.sum(totalLoss*totalLoss))
        # print(torch.sqrt(torch.sum(totalLoss*totalLoss)))
        # print(torch.sum(loss1 * loss1) / batch,torch.sum(loss2 * loss2) / batch,torch.sum(loss3 * loss3) / batch
        #       ,torch.sum(loss4 * loss4) / batch,torch.sum(disJointLoss * disJointLoss) / batch,torch.sum(negLoss * negLoss) / batch)
        return torch.sum(totalLoss*totalLoss) / batch





