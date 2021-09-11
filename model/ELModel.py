import torch.nn as nn
import torch
'''
Transformer the embedding to the left-bottom point of the box
'''
class CenterTransModel(nn.Module):
    def __init__(self, embedding_dim):
        super(CenterTransModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.Sigmoid(),
            nn.Linear(2 * embedding_dim, 4 * embedding_dim),
            nn.Sigmoid(),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.Sigmoid(),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self,input):
        return self.mlp(input)

'''
Transformer the embedding to the right-top point of box
'''
class OffsetTransModel(nn.Module):
    def __init__(self, embedding_dim):
        super(OffsetTransModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.Sigmoid(),
            nn.Linear(2 * embedding_dim, 4  * embedding_dim),
            nn.Sigmoid(),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.Sigmoid(),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
    def forward(self,input):
        return self.mlp(input)

'''
for intersection part
'''
class Deepset(nn.Module):
    def __init__(self, embedding_dim):
        super(Deepset, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.Tanh(),
            nn.Linear(2 * embedding_dim, 4 * embedding_dim),
            nn.Tanh(),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.Tanh(),
            nn.Linear(2 * embedding_dim, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.mlp(input)

'''
for intersection part
'''
class MLP4Intersection(nn.Module):
    def __init__(self, embedding_dim):
        super(MLP4Intersection, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.Tanh(),
            nn.Linear(2 * embedding_dim, 4 * embedding_dim),
            nn.Tanh(),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.Tanh(),
            nn.Linear(2 * embedding_dim, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.mlp(input)

class ELModel(nn.Module):
    '''

    Args:
        classNum: number of classes
        relationNum: number of relations
        embedding_dim: the dimension of the embedding(both class and relatio)
        margin: the distance that two box apart
    '''
    def __init__(self, device, classNum, relationNum, embedding_dim, margin=1e-9):
        super(ELModel, self).__init__()
        self.margin=margin
        self.classNum = classNum
        self.relationNum = relationNum
        self.device = device
        self.classEmbeddingDict = nn.Embedding(classNum,embedding_dim)
        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        self.centerTransModel = CenterTransModel(embedding_dim)
        self.offsetTransModel = OffsetTransModel(embedding_dim)

        self.deepset = Deepset(embedding_dim)
        self.mlp4intersection = MLP4Intersection(embedding_dim)


    #Limit length of vector normalized to 1
    #limit right-top is bigger than left-bottom
    def lengthShapeLoss(self):
        #classes length
        cClass = self.classEmbeddingDict(torch.tensor(range(self.classNum)).to(self.device))
        cClassCenter = self.centerTransModel(cClass)
        cClassOffset = self.offsetTransModel(cClass)
        classLengthLimit = torch.abs(torch.sum(cClassCenter * cClassCenter) - 1) / len(cClassCenter)
        classLengthLimit += torch.abs(torch.sum(cClassOffset * cClassOffset) - 1)/ len(cClassOffset)

        # classes box
        classBoxLimit = torch.sum(torch.maximum(cClassCenter - cClassOffset, torch.zeros(cClassOffset.shape,requires_grad=False).to(self.device))) / len(cClassOffset)

        # relation length
        cRelation = self.relationEmbeddingDict(torch.tensor(range(self.relationNum)).to(self.device))
        cRelationCenter = self.centerTransModel(cRelation)
        cRelationOffset = self.offsetTransModel(cRelation)
        relationLengthLimit = torch.abs(torch.sum(cRelationCenter * cRelationCenter) - 1) / len(cRelationCenter)
        relationLengthLimit += torch.abs(torch.sum(cRelationOffset * cRelationOffset) - 1) / len(cRelationOffset)

        # relation box
        relationBoxLimit = torch.sum(torch.maximum(cRelationCenter - cRelationOffset, torch.zeros(cRelationOffset.shape,requires_grad=False).to(self.device))) / len(
            cRelationOffset)

        return classLengthLimit + classBoxLimit + relationLengthLimit + relationBoxLimit


    # cClass isSubSetof dClass
    def nf1Loss(self,input):
        cClass = self.classEmbeddingDict(input[:,0])
        dClass = self.classEmbeddingDict(input[:,1])

        #get the center and offset of the box of the embedding
        cClassCenter = self.centerTransModel(cClass)
        cClassOffset = self.offsetTransModel(cClass)

        dClassCenter = self.centerTransModel(dClass)
        dClassOffset = self.offsetTransModel(dClass)

        margin = torch.ones(cClassCenter.shape,requires_grad=False)*self.margin
        margin = margin.to(self.device)

        leftBottomLimit =  torch.sum(torch.maximum(dClassCenter-cClassCenter, margin)) / len(cClassCenter)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset, margin)) / len(dClassOffset)

        return leftBottomLimit + righttopLimit

    def nf2Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])
        eClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = self.centerTransModel(cClass)
        cClassOffset = self.offsetTransModel(cClass)

        dClassCenter = self.centerTransModel(dClass)
        dClassOffset = self.offsetTransModel(dClass)

        eClassCenter = self.centerTransModel(eClass)
        eClassOffset = self.offsetTransModel(eClass)

        #get new center
        softmax = nn.Softmax(dim=1)
        softmax = softmax(torch.cat((self.mlp4intersection(cClass),self.mlp4intersection(dClass)),dim = 1))
        newCenter = cClassCenter*torch.unsqueeze(softmax[:,0],1) + dClassCenter*torch.unsqueeze(softmax[:,1],1)

        #get new offset
        theta = (self.deepset(cClass) + self.deepset(dClass)) / 2
        newOffset = torch.minimum(cClassOffset-cClassCenter,dClassOffset-dClassCenter) * theta
        newOffset = newCenter+newOffset

        # is subset
        margin = torch.ones(eClassCenter.shape, requires_grad=False) * self.margin
        margin = margin.to(self.device)

        leftBottomLimit = torch.sum(torch.maximum(eClassCenter - newCenter, margin)) / len(newCenter)
        righttopLimit = torch.sum(torch.maximum(newOffset - eClassOffset, margin)) / len(eClassOffset)

        return leftBottomLimit + righttopLimit
    def nf3Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        rRelation = self.relationEmbeddingDict(input[:, 1])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = self.centerTransModel(cClass)
        cClassOffset = self.offsetTransModel(cClass)

        dClassCenter = self.centerTransModel(dClass)
        dClassOffset = self.offsetTransModel(dClass)

        rRelationCenter = self.centerTransModel(rRelation)
        rRelationOffset = self.offsetTransModel(rRelation)

        #get new center
        cClassCenter = cClassCenter + rRelationCenter

        #get new offset
        tempOffset = (cClassOffset - cClassCenter) + (rRelationOffset - rRelationCenter)
        cClassOffset = cClassCenter + tempOffset

        # is subset
        margin = torch.ones(dClassCenter.shape, requires_grad=False) * self.margin
        margin = margin.to(self.device)

        leftBottomLimit = torch.sum(torch.maximum(dClassCenter - cClassCenter, margin)) / len(cClassCenter)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset, margin)) / len(dClassOffset)

        return leftBottomLimit + righttopLimit

    def nf4Loss(self, input):
        cClass = self.classEmbeddingDict(input[:, 1])
        rRelation = self.relationEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 2])

        cClassCenter = self.centerTransModel(cClass)
        cClassOffset = self.offsetTransModel(cClass)

        dClassCenter = self.centerTransModel(dClass)
        dClassOffset = self.offsetTransModel(dClass)

        rRelationCenter = self.centerTransModel(rRelation)
        rRelationOffset = self.offsetTransModel(rRelation)

        #get new center
        dClassCenter = dClassCenter + rRelationCenter

        #get new offset
        tempOffset = (dClassOffset - dClassCenter) + (rRelationOffset - rRelationCenter)
        dClassOffset = dClassCenter + tempOffset

        # is subset
        margin = torch.ones(dClassCenter.shape, requires_grad=False) * self.margin
        margin = margin.to(self.device)

        leftBottomLimit = torch.sum(torch.maximum(dClassCenter - cClassCenter, margin)) / len(cClassCenter)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset, margin)) / len(dClassOffset)

        return leftBottomLimit + righttopLimit

    def disJointLoss(self, input):
        cClass = self.classEmbeddingDict(input[:, 0])
        dClass = self.classEmbeddingDict(input[:, 1])


        cClassCenter = self.centerTransModel(cClass)
        cClassOffset = self.offsetTransModel(cClass)

        dClassCenter = self.centerTransModel(dClass)
        dClassOffset = self.offsetTransModel(dClass)


        # get new center
        softmax = nn.Softmax(dim=1)
        softmax = softmax(torch.cat((self.mlp4intersection(cClass), self.mlp4intersection(dClass)), dim=1))
        newCenter = cClassCenter * torch.unsqueeze(softmax[:, 0], 1) + dClassCenter * torch.unsqueeze(softmax[:, 1], 1)

        # get new offset
        theta = (self.deepset(cClass) + self.deepset(dClass)) / 2
        newOffset = torch.minimum(cClassOffset - cClassCenter, dClassOffset - dClassCenter) * theta
        newOffset = newCenter + newOffset

        # is subset
        margin = torch.zeros(newCenter.shape, requires_grad=False)
        margin = margin.to(self.device)

        rightLessLeftLoss = torch.sum(torch.maximum(newOffset - newCenter, margin)) / len(newCenter)


        return rightLessLeftLoss

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

        lengthShapeLoss = self.lengthShapeLoss()
        print(loss1.item(),loss2.item(),loss3.item(),loss4.item(),disJointLoss.item(),lengthShapeLoss.item())
        return loss1 + loss2 + loss3 + loss4 + disJointLoss + lengthShapeLoss









        


