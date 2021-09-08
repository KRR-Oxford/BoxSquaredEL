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
        self.device = device
        self.classEmbeddingDict = nn.Embedding(classNum,embedding_dim).to(device)
        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim).to(device)
        self.centerTransModel = CenterTransModel(embedding_dim).to(device)
        self.offsetTransModel = OffsetTransModel(embedding_dim).to(device)

    # cClass isSubSetof dClass
    def nf1Loss(self,input):


        cClass = self.classEmbeddingDict(input[:,0]).to(self.device)
        dClass = self.classEmbeddingDict(input[:,1]).to(self.device)

        #get the center and offset of the box of the embedding
        cClassCenter = self.centerTransModel(cClass).to(self.device)
        cClassOffset = self.offsetTransModel(cClass).to(self.device)

        dClassCenter = self.centerTransModel(dClass).to(self.device)
        dClassOffset = self.offsetTransModel(dClass).to(self.device)

        margin = torch.ones(cClassCenter.shape,requires_grad=False)*self.margin
        margin = margin.to(self.device)

        leftBottomLimit =  torch.sum(torch.maximum(dClassCenter-cClassCenter, margin)).to(self.device)
        righttopLimit = torch.sum(torch.maximum(cClassOffset - dClassOffset, margin)).to(self.device)

        #Todo:move to the outside, add only once
        boxLimit = torch.sum(torch.maximum(dClassCenter - dClassOffset, margin)).to(self.device)
        boxLimit +=torch.sum(torch.maximum(cClassCenter - cClassOffset, margin)).to(self.device)
        lengthLimit  = torch.abs(torch.sum(dClassCenter * dClassCenter)-1).to(self.device)
        lengthLimit += torch.abs(torch.sum(cClassCenter * cClassCenter) - 1).to(self.device)
        lengthLimit += torch.abs(torch.sum(cClassOffset * cClassOffset) - 1).to(self.device)
        lengthLimit += torch.abs(torch.sum(dClassOffset * dClassOffset) - 1).to(self.device)

        return leftBottomLimit + righttopLimit + boxLimit + lengthLimit





       # print(cClassCenter[0])
    def forward(self,input):

        nf1Data = input['nf1']
        nf1Data = nf1Data.to(self.device)
        loss1 = self.nf1Loss(nf1Data)
        return loss1









        


