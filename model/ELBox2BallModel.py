import numpy as np
import torch.nn as nn
import torch
np.random.seed(12)

class ELBox2BallModel(nn.Module):
    '''

    Args:
        classNum: number of classes
        relationNum: number of relations
        embedding_dim: the dimension of the embedding(both class and relatio)
        margin: the distance that two box apart
    '''

    def __init__(self, device, class_, relationNum, embedding_dim, margin=0,margin1=0):
        super(ELBox2BallModel, self).__init__()
        self.margin = margin
        self.margin1 = margin1
        self.classNum = len(class_)
        self.class_ = class_
        self.relationNum = relationNum
        self.device = device
        self.reg_norm=1
        self.inf=4

        self.classEmbeddingDict = nn.Embedding(self.classNum, embedding_dim*2)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight.data,axis=1).reshape(-1,1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim+1)
        nn.init.uniform_(self.relationEmbeddingDict.weight,a=-1,b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight.data, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim


    # cClass isSubSetof dClass
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

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 = torch.abs(d[:, self.embedding_dim:])

        # box

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin1).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False)).to(self.device)

        cen1 = c1
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(torch.maximum(euc + cr - dr+margin , zeros), axis=1), [-1, 1])
      #  print(cr)

        return dst+torch.reshape(torch.linalg.norm(torch.maximum(self.margin-cr, zeros), axis=1), [-1, 1])+\
               torch.reshape(torch.linalg.norm(torch.maximum( self.margin-dr, zeros), axis=1), [-1, 1])

    # cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])
        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]
        e1 = e[:, :self.embedding_dim]

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 =torch.abs( d[:, self.embedding_dim:])
        e2 = torch.abs(e[:, self.embedding_dim:])



        startAll = torch.maximum(c1-c2, d1-d2)
        endAll = torch.minimum(c1 + c2, d1 + d2)

        newR = (endAll-startAll)/2
        er = torch.abs(e2)

        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin1).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False)).to(self.device)

        cen1 =(startAll+endAll)/2
        cen2 = e1
        euc = torch.abs(cen1 - cen2)
        loss1 = torch.reshape(torch.linalg.norm(torch.maximum(startAll-endAll, zeros), axis=1), [-1, 1])

        dst = torch.reshape(torch.linalg.norm(torch.maximum(euc + newR - er +margin , zeros), axis=1), [-1, 1])\
              +loss1

       # print(torch.sum(loss1))
        return dst
    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 = torch.abs(d[:, self.embedding_dim:])

        # box

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin1).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False)).to(self.device)

        cen1 = c1
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(torch.maximum(-euc + cr + dr+margin , zeros), axis=1), [-1, 1])

        return dst+torch.reshape(torch.linalg.norm(torch.maximum(self.margin-cr, zeros), axis=1), [-1, 1])+\
               torch.reshape(torch.linalg.norm(torch.maximum( self.margin-dr, zeros), axis=1), [-1, 1])




    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])


        deltaR = r[:, -1].reshape(-1, 1)
        r = r[:, :-1]

        c1 = c[:, :self.embedding_dim]
        c2 = c[:, self.embedding_dim:]

        d1 = d[:, :self.embedding_dim]
        d2 = d[:, self.embedding_dim:]

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin1).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)

        cen1 = c1+r
        cen2 = d1
        euc = torch.abs(cen1-cen2)

        dst = torch.reshape(torch.linalg.norm(torch.maximum( euc+cr-dr +margin-torch.abs(deltaR) , zeros), axis=1),[-1,1])

        return  dst+torch.reshape(torch.linalg.norm(torch.maximum(self.margin-cr, zeros), axis=1), [-1, 1])+\
               torch.reshape(torch.linalg.norm(torch.maximum( self.margin-dr, zeros), axis=1), [-1, 1])+torch.abs(deltaR)



    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
       # r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 1])
        # deltaR = r[:, -1].reshape(-1,1)
        # r = r[:, :-1]

        c1 = c[:, :self.embedding_dim]
        c2 = c[:, self.embedding_dim:]

        d1 = d[:, :self.embedding_dim]
        d2 = d[:, self.embedding_dim:]

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin1).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False)).to(self.device)

        cen1 = c1
        cen2 = d1
        euc = torch.abs(cen1 - cen2)
        #print(deltaR.shape,euc.shape)

        dst = torch.reshape(torch.linalg.norm(torch.maximum(euc - cr - dr-margin , zeros), axis=1), [-1, 1])

        return dst+torch.reshape(torch.linalg.norm(torch.maximum(self.margin-cr, zeros), axis=1), [-1, 1])+\
               torch.reshape(torch.linalg.norm(torch.maximum( self.margin-dr, zeros), axis=1), [-1, 1])


    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])

        r = self.relationEmbeddingDict(input[:, 0])

        d = self.classEmbeddingDict(input[:, 2])
        deltaR = r[:, -1].reshape(-1, 1)
        r = r[:, :-1]

        c1 = c[:, :self.embedding_dim]
        c2 = c[:, self.embedding_dim:]

        d1 = d[:, :self.embedding_dim]
        d2 = d[:, self.embedding_dim:]

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin1).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False)).to(self.device)

        cen1 = c1 - r
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(torch.maximum(euc - cr - dr+margin+torch.abs(deltaR)  , zeros), axis=1), [-1, 1])

        return dst+torch.reshape(torch.linalg.norm(torch.maximum(self.margin-cr, zeros), axis=1), [-1, 1])+\
               torch.reshape(torch.linalg.norm(torch.maximum( self.margin-dr, zeros), axis=1), [-1, 1])+torch.abs(deltaR)








    def top_loss(self, input):
        d = self.classEmbeddingDict(input)

        d1 = d[:, :self.embedding_dim]

        d2 = torch.abs(d[:, self.embedding_dim:])


        dr = torch.abs(d2) / 2
        zeros = torch.zeros(dr.shape,requires_grad=False).to(self.device)
        dst = torch.reshape(torch.linalg.norm(torch.maximum(self.inf - dr, zeros), axis=1), [-1, 1])\
        +torch.reshape(torch.linalg.norm(torch.maximum(self.inf + d1, zeros), axis=1), [-1, 1])


        return dst



    def forward(self, input):
        batch = 512

        rand_index = np.random.choice(len(input['nf1']), size=batch)
        nf1Data = input['nf1'][rand_index]
        nf1Data = nf1Data.to(self.device)
        loss1 = self.nf1Loss(nf1Data)
        mseloss = nn.MSELoss(reduce=True)
        loss1 = mseloss(loss1, torch.zeros(loss1.shape, requires_grad=False).to(self.device))

        # nf2
        rand_index = np.random.choice(len(input['nf2']), size=batch)
        nf2Data = input['nf2'][rand_index]
        nf2Data = nf2Data.to(self.device)
        loss2 = self.nf2Loss(nf2Data)
        mseloss = nn.MSELoss(reduce=True)
        loss2 = mseloss(loss2, torch.zeros(loss2.shape, requires_grad=False).to(self.device))

        # nf3
        rand_index = np.random.choice(len(input['nf3']), size=batch)
        nf3Data = input['nf3'][rand_index]
        nf3Data = nf3Data.to(self.device)
        loss3 = self.nf3Loss(nf3Data)
        mseloss = nn.MSELoss(reduce=True)
        loss3 = mseloss(loss3, torch.zeros(loss3.shape, requires_grad=False).to(self.device))

        # nf4
        rand_index = np.random.choice(len(input['nf4']), size=batch)
        nf4Data = input['nf4'][rand_index]
        nf4Data = nf4Data.to(self.device)
        loss4 = self.nf4Loss(nf4Data)
        mseloss = nn.MSELoss(reduce=True)
        loss4 = mseloss(loss4, torch.zeros(loss4.shape, requires_grad=False).to(self.device))

        # disJoint
        rand_index = np.random.choice(len(input['disjoint']), size=batch)
        disJointData = input['disjoint'][rand_index]
        disJointData = disJointData.to(self.device)
        disJointLoss = self.disJointLoss(disJointData)
        mseloss = nn.MSELoss(reduce=True)
        disJointLoss = mseloss(disJointLoss, torch.zeros(disJointLoss.shape, requires_grad=False).to(self.device))

        # top_loss
        rand_index = np.random.choice(len(input['top']), size=batch)
        topData = input['top'][rand_index]
        topData = topData.to(self.device)
        topLoss = self.top_loss(topData)
        mseloss = nn.MSELoss(reduce=True)
        topLoss = mseloss(topLoss, torch.ones(topLoss.shape, requires_grad=False).to(self.device) * 2)

        # negLoss
        rand_index = np.random.choice(len(input['nf3_neg']), size=batch)
        negData = input['nf3_neg'][rand_index]
        negData = negData.to(self.device)
        negLoss = self.neg_loss(negData)
        #print('negLoss',negLoss.sum().item() / batch)
        mseloss = nn.MSELoss(reduce=True)
        negLoss = mseloss(negLoss, torch.ones(negLoss.shape, requires_grad=False).to(self.device)*1.5)

        # print(loss1,loss2,loss3, loss4,disJointLoss,negLoss)
     #   print(  loss3, loss4,   negLoss)
#  loss1+loss2+disJointLoss+
        totalLoss = loss1+loss2#+disJointLoss+loss3+loss4#+topLoss##+ negLoss  # +negLoss #loss4 +disJointLoss+loss1 + loss2 +  negLoss#+ disJointLoss+ topLoss+ loss3 + loss4 +  negLoss

        return (totalLoss)

#human 50 -0.1 1 0.08 0.50 282.43 0.95
#human 50 -0.1 1 0.19 0.67 236.05 0.96