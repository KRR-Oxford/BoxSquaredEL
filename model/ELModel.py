import torch.nn as nn
import torch

# '''
# Transformer the embedding to the left-bottom point of the box
# '''
# class CenterTransModel(nn.Module):
#     def __init__(self, embedding_dim):
#         super(CenterTransModel, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 4 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(4 * embedding_dim, 8 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(8 * embedding_dim, 4 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(4 * embedding_dim, embedding_dim),
#             nn.Sigmoid(),
#         )
#
#     def forward(self,input):
#         return self.mlp(input)*5
#
# '''
# Transformer the embedding to the right-top point of box
# '''
# class OffsetTransModel(nn.Module):
#     def __init__(self, embedding_dim):
#         super(OffsetTransModel, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 4 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(4 * embedding_dim, 8  * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(8 * embedding_dim, 4 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(4 * embedding_dim, embedding_dim),
#             nn.Sigmoid(),
#         )
#     def forward(self,input):
#         return self.mlp(input)*5
#
# '''
# for intersection part
# '''
# class Deepset(nn.Module):
#     def __init__(self, embedding_dim):
#         super(Deepset, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 2 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(2 * embedding_dim, 4 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(4 * embedding_dim, 2 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(2 * embedding_dim, 1),
#             nn.Tanh(),
#             nn.Dropout(0.3)
#         )
#
#     def forward(self, input):
#         return self.mlp(input)
#
# '''
# for intersection part
# '''
# class MLP4Intersection(nn.Module):
#     def __init__(self, embedding_dim):
#         super(MLP4Intersection, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 2 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(2 * embedding_dim, 4 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(4 * embedding_dim, 2 * embedding_dim),
#             nn.Tanh(),
#             nn.Linear(2 * embedding_dim, 1),
#             nn.Tanh(),
#             nn.Dropout(0.3)
#         )
#
#     def forward(self, input):
#         return self.mlp(input)

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
        self.relationNum = relationNum
        self.device = device
        self.classEmbeddingDict = nn.Embedding(classNum,embedding_dim*2)
        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim*2)
        self.embedding_dim = embedding_dim
        # self.centerTransModel = CenterTransModel(embedding_dim)
        # self.offsetTransModel = OffsetTransModel(embedding_dim)
        #
        # self.deepset = Deepset(embedding_dim)
        # self.mlp4intersection = MLP4Intersection(embedding_dim)

    # cClass isSubSetof dClass
    def nf1Loss(self,input):

        cClass = self.classEmbeddingDict(input[:,0])
        dClass = self.classEmbeddingDict(input[:,1])

        #get the center and offset of the box of the embedding
        cClassCenter = cClass[:,0:self.embedding_dim]
        cClassOffset = cClass[:,self.embedding_dim:2*self.embedding_dim]

        dClassCenter = dClass[:,0:self.embedding_dim]
        dClassOffset = dClass[:,self.embedding_dim:2*self.embedding_dim]

        margin = (torch.ones(cClassOffset.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(cClassOffset.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit =  torch.sum(torch.maximum(dClassCenter-cClassCenter+margin, zeros)) / len(cClassCenter)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset+margin, zeros)) / len(dClassOffset)

        shapeLimit = torch.sum(torch.maximum(cClassCenter - cClassOffset+margin,
                                                     zeros)) / len(cClassOffset)

        shapeLimit += torch.sum(torch.maximum(dClassCenter - dClassOffset+margin,
                                                     zeros)) / len(dClassOffset)
        print(leftBottomLimit,righttopLimit,shapeLimit)
        return leftBottomLimit + righttopLimit + shapeLimit

    #cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])
        eClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, 0:self.embedding_dim]
        cClassOffset = cClass[:, self.embedding_dim:2 * self.embedding_dim]

        dClassCenter = dClass[:, 0:self.embedding_dim]
        dClassOffset = dClass[:, self.embedding_dim:2 * self.embedding_dim]


        eClassCenter = eClass[:, 0:self.embedding_dim]
        eClassOffset = eClass[:, self.embedding_dim:2 * self.embedding_dim]

        startAll = torch.maximum(cClassCenter, dClassCenter)
        endAll = torch.minimum(cClassOffset, dClassOffset)

        margin = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)
        # print(endX,startX)



        leftBottomLimit = torch.sum(torch.maximum(eClassCenter - startAll+margin,zeros)) / len(eClassCenter)

        righttopLimit = torch.sum(torch.maximum(endAll - eClassOffset + margin,zeros)) / len(eClassOffset)

        shapeLimit = torch.sum(torch.maximum(cClassCenter - cClassOffset+margin,
                                             zeros)) / len(cClassOffset)

        shapeLimit += torch.sum(torch.maximum(dClassCenter - dClassOffset+margin,
                                             zeros)) / len(dClassOffset)
        shapeLimit += torch.sum(torch.maximum(eClassCenter - eClassOffset+margin,
                                             zeros)) / len(eClassOffset)

        # # get new center
        # softmax = nn.Softmax(dim=1)
        # softmax = softmax(torch.cat((self.mlp4intersection(cClass), self.mlp4intersection(dClass)), dim=1))
        # newCenter = (cClassCenter + cClassOffset) / 2 * torch.unsqueeze(softmax[:, 0], 1) + (
        #             dClassCenter + dClassOffset) / 2 * torch.unsqueeze(softmax[:, 1], 1)
        #
        # # get new offset
        # theta = (self.deepset(cClass) + self.deepset(dClass)) / 2
        # newOffset = torch.minimum(cClassOffset - cClassCenter, dClassOffset - dClassCenter) * theta
        # newCenter = newCenter - newOffset / 2
        # newOffset = newCenter + newOffset
        #
        # # is subset
        # margin = torch.ones(eClassCenter.shape, requires_grad=False) * self.margin
        # margin = margin.to(self.device)
        #
        # leftBottomLimit = torch.sum(torch.maximum(eClassCenter - newCenter, margin)) / len(newCenter)
        # righttopLimit = torch.sum(torch.maximum(newOffset - eClassOffset, margin)) / len(eClassOffset)

        return leftBottomLimit + righttopLimit + shapeLimit

    # cClass isSubSet of relation some dClass
    def nf3Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        rRelation = self.relationEmbeddingDict(input[:, 1])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, 0:self.embedding_dim]
        cClassOffset = cClass[:, self.embedding_dim:2 * self.embedding_dim]

        dClassCenter = dClass[:, 0:self.embedding_dim]
        dClassOffset = dClass[:, self.embedding_dim:2 * self.embedding_dim]

        rRelationCenter = rRelation[:, 0:self.embedding_dim]
        rRelationOffset = rRelation[:, self.embedding_dim:2 * self.embedding_dim]


        #get new center
        cClassCenter = cClassCenter + rRelationCenter

        #get new offset
        tempOffset = (cClassOffset - cClassCenter) + (rRelationOffset - rRelationCenter)
        cClassOffset = cClassCenter + tempOffset

        # is subset
        margin = (torch.ones(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.sum(torch.maximum(dClassCenter - cClassCenter+ margin,zeros)) / len(cClassCenter)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset+ margin,zeros)) / len(dClassOffset)

        shapeLimit = torch.sum(torch.maximum(dClassCenter - dClassOffset+ margin,zeros)) / len(dClassOffset)

        shapeLimit += torch.sum(torch.maximum(cClassCenter - cClassOffset+ margin,zeros)) / len(cClassOffset)

        return leftBottomLimit + righttopLimit

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 1])
        rRelation = self.relationEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = cClass[:, 0:self.embedding_dim]
        cClassOffset = cClass[:, self.embedding_dim:2 * self.embedding_dim]

        dClassCenter = dClass[:, 0:self.embedding_dim]
        dClassOffset = dClass[:, self.embedding_dim:2 * self.embedding_dim]

        rRelationCenter = rRelation[:, 0:self.embedding_dim]
        rRelationOffset = rRelation[:, self.embedding_dim:2 * self.embedding_dim]

        #get new center
        dClassCenter = dClassCenter + rRelationCenter
        #get new offset
        tempOffset = (dClassOffset - dClassCenter) + (rRelationOffset - rRelationCenter)
        dClassOffset = dClassCenter + tempOffset

        # is subset
        margin = (torch.ones(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(dClassCenter.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.sum(torch.maximum(dClassCenter - cClassCenter+ margin,zeros)) / len(cClassCenter)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset+ margin,zeros)) / len(dClassOffset)

        return leftBottomLimit + righttopLimit

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
        # startX = torch.maximum(cClassCenter[:, 0], dClassCenter[:, 0])
        # endX = torch.minimum(cClassOffset[:, 0], dClassOffset[:, 0])
        #
        # startY = torch.maximum(cClassCenter[:, 1], dClassCenter[:, 1])
        # endY = torch.minimum(cClassOffset[:, 1], dClassOffset[:, 1])
        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)
       # print(endX,startX)

        rightLessLeftLoss = torch.sum((torch.maximum(endAll - startAll+margin,zeros)))/len(endAll)






        # # get new center
        # softmax = nn.Softmax(dim=1)
        # softmax = softmax(torch.cat((self.mlp4intersection(cClass), self.mlp4intersection(dClass)), dim=1))
        # newCenter = (cClassCenter+cClassOffset)/2 * torch.unsqueeze(softmax[:, 0], 1) + (dClassCenter+dClassOffset)/2 * torch.unsqueeze(softmax[:, 1], 1)
        #
        # # get new offset
        # theta = (self.deepset(cClass) + self.deepset(dClass)) / 2
        # newOffset = torch.minimum(cClassOffset - cClassCenter, dClassOffset - dClassCenter) * theta
        # newCenter = newCenter - newOffset/2
        # newOffset = newCenter + newOffset/2
        # print(softmax)
        # #print(cClassCenter, cClassOffset, dClassCenter,dClassOffset, newCenter, newOffset )
        #
        # # is subset
        # margin = (torch.ones(newCenter.shape, requires_grad=False)*self.margin).to(self.device)
        # zeros  = (torch.zeros(newCenter.shape, requires_grad=False)*self.margin).to(self.device)
        #
        # rightLessLeftLoss = torch.sum(torch.maximum(newOffset - newCenter,zeros )) / len(newCenter)

        shapeLoss = torch.sum(torch.maximum(cClassCenter - cClassOffset+margin,
                                                zeros)) / len(cClassOffset)

        shapeLoss += torch.sum(torch.maximum(dClassCenter - dClassOffset+margin,
                                                     zeros)) / len(dClassOffset)

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


    def forward(self,input):


        # nf1
        nf1Data = input['nf1']
        nf1Data = nf1Data.to(self.device)
        loss1 = self.nf1Loss(nf1Data)

        #nf2
        nf2Data = input['nf2']
        nf2Data = nf2Data.to(self.device)
        loss2 = self.nf2Loss(nf2Data)

        # nf3
        nf3Data = input['nf3']
        nf3Data = nf3Data.to(self.device)
        loss3 = self.nf3Loss(nf3Data)

        # nf4
        nf4Data = input['nf4']
        nf4Data = nf4Data.to(self.device)
        loss4 = self.nf4Loss(nf4Data)

        # disJoint
        disJointData = input['disjoint']
        disJointData = disJointData.to(self.device)
        disJointLoss = self.disJointLoss(disJointData)

        # top_loss
        topData = input['top']
        topData = topData.to(self.device)
        topLoss = self.top_loss(topData)


        print(loss1.item(),loss2.item(),loss3.item(),loss4.item(),disJointLoss.item(),topLoss.item() )
        #  print( loss1,loss2,disJointLoss)

        return  loss1 + loss2 + loss3 + loss4 + disJointLoss + topLoss

 







        


