import numpy as np
import torch.nn as nn
import torch

class ELBallModel(nn.Module):
    '''

    Args:
        classNum: number of classes
        relationNum: number of relations
        embedding_dim: the dimension of the embedding(both class and relatio)
        margin: the distance that two box apart
    '''

    def __init__(self, device, classNum, relationNum, embedding_dim, margin=0):
        super(ELBallModel, self).__init__()
        self.margin = margin
        self.classNum = classNum
        self.relationNum = relationNum
        self.device = device

        self.classEmbeddingDict = nn.Embedding(classNum, embedding_dim+1)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight,axis=1).reshape(-1,1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        nn.init.uniform_(self.relationEmbeddingDict.weight,a=-1,b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim

    # cClass isSubSetof dClass
    def nf1Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])

        # get the center and offset of the box of the embedding
        cClassCenter = cClass[:, :self.embedding_dim]
        cClassRadiu = torch.abs(cClass[:, -1])

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassRadiu = torch.abs(dClass[:, -1])

        zeros = (torch.zeros(dClassRadiu.shape, requires_grad=False) ).to(self.device)
        margin= (torch.ones(dClassRadiu.shape, requires_grad=False)*self.margin ).to(self.device)
        #print(torch.linalg.norm(cClassCenter-dClassCenter,axis=1).shape, cClassRadiu.shape)
        relu = torch.nn.ReLU()
        loss1 = relu(torch.linalg.norm(cClassCenter-dClassCenter,axis=1)+cClassRadiu-dClassRadiu-margin)

        loss2 = torch.abs(torch.linalg.norm(cClassCenter,axis=1)-1)+torch.abs(torch.linalg.norm(dClassCenter,axis=1)-1)

        return torch.sum(torch.sqrt(loss1 + loss2))/len(cClass)
    # cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])
        eClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, :self.embedding_dim]
        cClassRadiu = torch.abs(cClass[:, -1])

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassRadiu = torch.abs(dClass[:, -1])

        eClassCenter = eClass[:, :self.embedding_dim]
        eClassRadiu = torch.abs(eClass[:, -1])


        zeros = (torch.zeros(dClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)
        margin = (torch.ones(dClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)

        relu = torch.nn.ReLU()


        loss1 = relu(torch.linalg.norm(cClassCenter-dClassCenter,axis=1)-cClassRadiu-dClassRadiu-margin)\
              + relu(torch.linalg.norm(cClassCenter - eClassCenter,axis=1)-cClassRadiu-margin) \
              + relu(torch.linalg.norm(dClassCenter - eClassCenter,axis=1) - dClassRadiu - margin)
            #  + torch.maximum(zeros, torch.minimum(cClassRadiu,dClassRadiu)-eClassRadiu- margin )

        loss2 = torch.abs(torch.linalg.norm(cClassCenter,axis=1)-1)+torch.abs(torch.linalg.norm(dClassCenter,axis=1)-1)+torch.abs(torch.linalg.norm(eClassCenter,axis=1)-1)





        return torch.sum(torch.sqrt(loss1 + loss2))/len(cClass)

    # cClass isSubSet of relation some dClass
    def nf3Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        rRelation = self.relationEmbeddingDict(input[:, 1])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, :self.embedding_dim]
        cClassRadiu = torch.abs(cClass[:, -1])

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassRadiu  = torch.abs(dClass[:, -1])

        rRelationCenter = rRelation[:, :self.embedding_dim]


        # is subset
        zeros = (torch.zeros(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)
        margin = (torch.ones(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)

        loss1 = torch.maximum(zeros,
                              torch.linalg.norm(cClassCenter +rRelationCenter- dClassCenter,axis=1) + cClassRadiu - dClassRadiu - margin)

        loss2 = torch.abs(torch.linalg.norm(cClassCenter,axis=1) - 1) + torch.abs(torch.linalg.norm(dClassCenter,axis=1) - 1)

        return torch.sum(torch.sqrt(loss1 + loss2))/len(cClass)

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 1])

        rRelation = self.relationEmbeddingDict(input[:, 0])

        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, :self.embedding_dim]
        cClassRadiu = torch.abs(cClass[:, -1])

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassRadiu = torch.abs(dClass[:, -1])

        rRelationCenter = rRelation[:, :self.embedding_dim]

        # is subset
        zeros = (torch.zeros(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)
        margin = (torch.ones(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)

        loss1 = torch.maximum(zeros,
                              torch.linalg.norm(
                                  cClassCenter - rRelationCenter - dClassCenter,axis=1) - cClassRadiu - dClassRadiu - margin)

        loss2 = torch.abs(torch.linalg.norm(cClassCenter,axis=1) - 1) + torch.abs(torch.linalg.norm(dClassCenter,axis=1) - 1)

        return torch.sum(torch.sqrt(loss1 + loss2))/len(cClass)
    def disJointLoss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])

        cClassCenter = cClass[:, :self.embedding_dim]
        cClassRadiu  = torch.abs(cClass[:, -1])

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassRadiu  = torch.abs(dClass[:, -1])
        # print(cClassCenter,cClassRadiu)
        # print(dClassCenter,dClassRadiu)

        # mathematical method
        zeros = (torch.zeros(cClassRadiu.shape, requires_grad=False)).to(self.device)
        margin = (torch.ones(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)

        loss1 = torch.maximum(zeros,
                              -torch.linalg.norm(
                                  cClassCenter - dClassCenter,axis=1) + cClassRadiu + dClassRadiu + margin)

        loss2 = torch.abs(torch.linalg.norm(cClassCenter,axis=1) - 1) + torch.abs(torch.linalg.norm(dClassCenter,axis=1) - 1)

        return torch.sum(torch.sqrt(loss1 + loss2))/len(cClass)


    def top_loss(self, input):
        top = self.classEmbeddingDict(input[0])
        topCenter = top[0:self.embedding_dim]
        topRadiu = torch.abs(top[-1])

        # infinity
        inf = 1

        return torch.maximum(torch.tensor(0),inf-topRadiu)

    # cClass is_NOT_SubSet of relation some dClass
    def neg_loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        rRelation = self.relationEmbeddingDict(input[:, 1])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, :self.embedding_dim]
        cClassRadiu = torch.abs(cClass[:, -1])

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassRadiu = torch.abs(dClass[:, -1])

        rRelationCenter = rRelation[:, :self.embedding_dim]

        # is subset
        zeros = (torch.zeros(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)
        margin = (torch.ones(cClassRadiu.shape, requires_grad=False) * self.margin).to(self.device)

        loss1 = torch.sum(torch.maximum(zeros,
                              -torch.linalg.norm(
                                  cClassCenter +rRelationCenter - dClassCenter,axis=1) +cClassRadiu+ dClassRadiu + margin))

        loss2 = torch.abs(torch.linalg.norm(cClassCenter,axis=1) - 1) + torch.abs(torch.linalg.norm(dClassCenter,axis=1) - 1)

        return torch.sum(torch.sqrt(loss1 + loss2))/len(cClass)


    def forward(self, input):
        batch = 128
        #print(input['disjoint'])
        # nf1
        rand_index = np.random.choice(len(input['nf1']), size=batch)

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

        print(loss1.item(), loss2.item(), loss3.item(), loss4.item(), disJointLoss.item(), topLoss.item(),
              negLoss.item())
        #  print( loss1,loss2,disJointLoss)

        return loss1 + loss2 + loss3 + loss4 + disJointLoss +  negLoss
