import torch.nn as nn
import torch
from TransR import TransR
import numpy as np



class ELModel(nn.Module):
    '''

    Args:
        classNum: number of classes
        relationNum: number of relations
        embedding_dim: the dimension of the embedding(both class and relatio)
        margin: the distance that two box apart
    '''
    def __init__(self, device, classNum, relationNum, embedding_dim, margin=0):
        super(ELModel, self).__init__()
        self.margin=margin
        self.classNum = classNum
        self.TransR = TransR(embedding_dim)
        self.relationNum = relationNum
        self.device = device
        self.inf = 5
        self.reg_norm = 1
        self.classEmbeddingDict = nn.Embedding(classNum, embedding_dim*2)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=0, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight, axis=1).reshape(-1, 1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        nn.init.uniform_(self.relationEmbeddingDict.weight, a=0, b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim
        # self.centerTransModel = CenterTransModel(embedding_dim)
        # self.offsetTransModel = OffsetTransModel(embedding_dim)
        #
        # self.deepset = Deepset(embedding_dim)
        # self.mlp4intersection = MLP4Intersection(embedding_dim)

    # cClass isSubSetof dClass
    def reg(self, x):
        res = torch.abs(torch.linalg.norm(x, axis=1) - self.reg_norm)
        res = torch.reshape(res, [-1, 1])
        return res

    def nf1Loss(self,input):

        cClass = self.classEmbeddingDict(input[:,0])
        dClass = self.classEmbeddingDict(input[:,1])

        #get the center and offset of the box of the embedding
        cClassCenter = cClass[:, :self.embedding_dim]
        cClassOffset = cClass[:, self.embedding_dim:]

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassOffset = dClass[:, self.embedding_dim:]
       # print(cClassOffset.shape, dClassCenter.shape)

        margin = (torch.ones(cClassOffset.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(cClassOffset.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit =  torch.linalg.norm(torch.maximum(dClassCenter-cClassCenter+margin, zeros), axis=1)
        righttopLimit = torch.linalg.norm(torch.maximum(cClassOffset - dClassOffset+margin, zeros), axis=1)

        shapeLimit = torch.linalg.norm(torch.maximum(cClassCenter - cClassOffset+margin,
                                                     zeros), axis =1)

        shapeLimit += torch.linalg.norm(torch.maximum(dClassCenter - dClassOffset+margin,
                                                     zeros), axis=1)
       # print(shapeLimit.shape)
        #print(leftBottomLimit,righttopLimit,shapeLimit)
        return leftBottomLimit + righttopLimit + shapeLimit

    #cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])
        eClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, :self.embedding_dim]
        cClassOffset = cClass[:, self.embedding_dim:]

        dClassCenter = dClass[:, :self.embedding_dim]
        dClassOffset = dClass[:, self.embedding_dim:]


        eClassCenter = eClass[:, :self.embedding_dim]
        eClassOffset = eClass[:, self.embedding_dim:]

        startAll = torch.maximum(cClassCenter, dClassCenter)
        endAll = torch.minimum(cClassOffset, dClassOffset)

        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)
        #print(margin.shape)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.linalg.norm(torch.maximum(eClassCenter - startAll+margin,zeros), axis=1)
        righttopLimit = torch.linalg.norm(torch.maximum(endAll - eClassOffset + margin,zeros),axis=1)

        shapeLimit = torch.linalg.norm(torch.maximum(cClassCenter - cClassOffset+margin,
                                             zeros),axis=1)

        shapeLimit += torch.linalg.norm(torch.maximum(dClassCenter - dClassOffset+margin,
                                             zeros),axis=1)
        shapeLimit += torch.linalg.norm(torch.maximum(eClassCenter - eClassOffset+margin,
                                             zeros),axis=1)


        return leftBottomLimit + righttopLimit + shapeLimit

    # cClass isSubSet of relation some dClass
    def nf3Loss(self, input):

        cClass = self.classEmbeddingDict(input[:, 0])
        rRelation = self.relationEmbeddingDict(input[:, 1])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = self.TransR(cClass[:, :self.embedding_dim])
        cClassOffset = self.TransR(cClass[:, self.embedding_dim:])

        dClassCenter = self.TransR(dClass[:, :self.embedding_dim])
        dClassOffset = self.TransR(dClass[:, self.embedding_dim:])

        rRelationCenter = rRelation[:, :self.embedding_dim]



        #get new center

        cClassCenter = cClassCenter + rRelationCenter

        #get new offset
        cClassOffset = cClassOffset + rRelationCenter

        # is subset
        margin = (torch.ones(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.linalg.norm(torch.maximum(dClassCenter - cClassCenter+ margin,zeros),axis=1)
        righttopLimit = torch.linalg.norm(torch.maximum(cClassOffset - dClassOffset+ margin,zeros),axis=1)
        shapeLimit = torch.linalg.norm(torch.maximum(dClassCenter - dClassOffset+ margin,zeros),axis=1)

        shapeLimit += torch.linalg.norm(torch.maximum(cClassCenter - cClassOffset+ margin,zeros),axis=1)

        return leftBottomLimit + righttopLimit + shapeLimit

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 1])
        rRelation = self.relationEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 2])


        cClassCenter = self.TransR(cClass[:, :self.embedding_dim])
        cClassOffset = self.TransR(cClass[:, self.embedding_dim:])

        dClassCenter = self.TransR(dClass[:, :self.embedding_dim])
        dClassOffset = self.TransR(dClass[:, self.embedding_dim:])

        rRelationCenter = rRelation[:, 0:self.embedding_dim]

        #get new center
        dClassCenter = dClassCenter + rRelationCenter
        #get new offset
        dClassOffset = dClassOffset + rRelationCenter

        # is subset
        margin = (torch.ones(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.linalg.norm(torch.maximum(dClassCenter - cClassCenter+ margin,zeros),axis=1)
        righttopLimit = torch.linalg.norm(torch.maximum(cClassOffset - dClassOffset+ margin,zeros),axis=1)

        shapeLimit = torch.linalg.norm(torch.maximum(dClassCenter - dClassOffset + margin, zeros),axis=1)

        shapeLimit += torch.linalg.norm(torch.maximum(cClassCenter - cClassOffset + margin, zeros),axis=1)

        return leftBottomLimit + righttopLimit + shapeLimit

    def disJointLoss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])

        cClassCenter = cClass[:, 0:self.embedding_dim]
        cClassOffset = cClass[:, self.embedding_dim:2 * self.embedding_dim]

        dClassCenter = dClass[:, 0:self.embedding_dim]
        dClassOffset = dClass[:, self.embedding_dim:2 * self.embedding_dim]


        # mathematical method
        startAll = torch.maximum(cClassCenter, dClassCenter)
        endAll   = torch.minimum(cClassOffset, dClassOffset)

        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        rightLessLeftLoss = torch.linalg.norm(torch.maximum(endAll - startAll + margin,zeros),axis=1)

        shapeLoss = torch.linalg.norm(torch.maximum(cClassCenter - cClassOffset+margin,
                                                zeros),axis=1)

        shapeLoss += torch.linalg.norm(torch.maximum(dClassCenter - dClassOffset+margin,
                                                     zeros),axis=1)

        return rightLessLeftLoss+ shapeLoss

    def top_loss(self, input):
        top = self.classEmbeddingDict(input[0])
       # print('top',top)
       # print(top,'top')
        topCenter = top[0:self.embedding_dim]
        topOffset = top[self.embedding_dim:2 * self.embedding_dim]



        #infinity
        inf = 1
        margin =   (torch.ones(topOffset.shape, requires_grad=False).to(self.device) )* inf
        zeros = (torch.zeros(topOffset.shape, requires_grad=False).to(self.device)) * inf
        return torch.sum(torch.maximum( topCenter-topOffset+ margin,zeros))

    # cClass is_NOT_SubSet of relation some dClass
    def neg_loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        rRelation = self.relationEmbeddingDict(input[:, 1])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = self.TransR(cClass[:, :self.embedding_dim])
        cClassOffset = self.TransR(cClass[:, self.embedding_dim:])

        dClassCenter = self.TransR(dClass[:, :self.embedding_dim])
        dClassOffset = self.TransR(dClass[:, self.embedding_dim:])

        rRelationCenter = rRelation[:, 0:self.embedding_dim]

        # get new center
        cClassCenter = cClassCenter + rRelationCenter

        # get new offset
        cClassOffset = cClassOffset + rRelationCenter

        startAll = torch.maximum(cClassCenter, dClassCenter)
        endAll = torch.minimum(cClassOffset, dClassOffset)

        andBox = endAll-startAll
        cBox   = cClassOffset - cClassCenter

       # negLoss = torch.max(torch.tensor(0).to(self.device), andBox-cBox)

        # is subset

        margin = (torch.ones(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)

        negLoss = torch.linalg.norm(torch.maximum(-dClassCenter + cClassCenter - margin, zeros), axis=1)
        negLoss += torch.linalg.norm(torch.maximum(-cClassOffset + dClassOffset - margin, zeros), axis=1)
        shapeLimit = torch.linalg.norm(torch.maximum(dClassCenter - dClassOffset + margin, zeros), axis=1)

        shapeLimit += torch.linalg.norm(torch.maximum(cClassCenter - cClassOffset + margin, zeros), axis=1)



        return negLoss + shapeLimit

    def forward(self, input):
        batch = 256
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

        totalLoss = loss1 + loss2 + loss3 + loss4 + disJointLoss #+ negLoss  # + topLoss

        # print(torch.sum(totalLoss*totalLoss))
        # print(torch.sqrt(torch.sum(totalLoss*totalLoss)))
        return torch.sum(totalLoss * totalLoss) / batch


        


