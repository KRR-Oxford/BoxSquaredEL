import torch.nn as nn
import torch
from model.TransR import TransR
import numpy as np

class RelationModel(nn.Module):
    def __init__(self, embedding_dim):
        super(RelationModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, 8  * embedding_dim),
            nn.ReLU(),
            nn.Linear(8 * embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.ReLU(),
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
        # self.centerTransModel = CenterTransModel(embedding_dim)
        # self.offsetTransModel = OffsetTransModel(embedding_dim)
        #
        # self.deepset = Deepset(embedding_dim)
        # self.mlp4intersection = MLP4Intersection(embedding_dim)


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
        rc = torch.linalg.norm(c2-c1,axis=1)/2
        rd = torch.linalg.norm(d2-d1,axis=1)/2
        x1 = (c1 + c2) / 2
        x2 = (d1 + d2) / 2
        euc = torch.linalg.norm(x1 - x2, axis=1)
        relu = torch.nn.ReLU()
        dst = torch.reshape(relu(euc + rc - rd + self.margin), [-1, 1])

        # box

        margin = (torch.ones(c1.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(c1.shape, requires_grad=False) * self.margin).to(self.device)
        relu1 = nn.ReLU()
        leftBottomLimit = torch.reshape(torch.linalg.norm(relu1(d1 - c1 + margin), axis=1), [-1, 1])
        relu2 = nn.ReLU()
        righttopLimit = torch.reshape(torch.linalg.norm(relu2(c2 - d2 + margin), axis=1), [-1, 1])

        relu3 = nn.ReLU()
        shapeLimit = torch.reshape(torch.linalg.norm(relu3(c1 - c2
                                                     ), axis=1,ord=1), [-1, 1])
        relu4 = nn.ReLU()
        shapeLimit += torch.reshape(torch.linalg.norm(relu4(d1 - d2
                                                      ), axis=1, ord = 1), [-1, 1])
        return leftBottomLimit+righttopLimit+ shapeLimit + self.reg(c1) + self.reg(c2) + self.reg(d1) + self.reg(d2)

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
      #  print(torch.linalg.norm(c2 - c1, axis=1).shape,torch.reshape(torch.linalg.norm(c2 - c1, axis=1) / 2, [-1, 1]).shape)
      #   rc = torch.linalg.norm(c2 - c1, axis=1) / 2
      #   rd = torch.linalg.norm(d2 - d1, axis=1) / 2
      #   re = torch.linalg.norm(e2 - e1, axis=1) / 2

        rc = torch.reshape(torch.linalg.norm(c2 - c1, axis=1) / 2, [-1, 1])
        rd = torch.reshape(torch.linalg.norm(d2 - d1, axis=1) / 2, [-1, 1])
        re = torch.reshape(torch.linalg.norm(e2 - e1, axis=1) / 2, [-1, 1])
        x1 = (c1 + c2) / 2
        x2 = (d1 + d2) / 2
        x3 = (e1 + e2) / 2


        #
        # c1 = c[:, :self.embedding_dim]
        # d1 = d[:, :self.embedding_dim]
        # e1 = e[:, :self.embedding_dim]
        #
        # rc = torch.reshape(torch.abs(c[:, -1]), [-1, 1])
        # rd = torch.reshape(torch.abs(d[:, -1]), [-1, 1])
        # re = torch.reshape(torch.abs(e[:, -1]), [-1, 1])





        sr = rc + rd

        x = x2 - x1
        dst = torch.reshape(torch.linalg.norm(x, axis=1), [-1, 1])
        dst2 = torch.reshape(torch.linalg.norm(x3 - x1, axis=1), [-1, 1])
        dst3 = torch.reshape(torch.linalg.norm(x3 - x2, axis=1), [-1, 1])
        relu = torch.nn.ReLU()
        rdst = relu(torch.minimum(rc, rd) - re)
        relu1 = torch.nn.ReLU()
        relu2 = torch.nn.ReLU()
        relu3 = torch.nn.ReLU()
        relu4 = torch.nn.ReLU()
        relu5 = torch.nn.ReLU()


        dst_loss = (relu1(dst - sr)
                    + relu2(dst2 - rc)
                    + relu3(dst3 - rd)
                    + rdst - self.margin)

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c2, d2)
        endAll = torch.maximum(endAll-startAll,torch.zeros(startAll.shape))+startAll

        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)
        # print(margin.shape)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.reshape(torch.linalg.norm(relu1(e1 - startAll ), axis=1),[-1,1])
        righttopLimit = torch.reshape(torch.linalg.norm(relu2(endAll - e2 ), axis=1),[-1,1])

        shapeLimit = torch.reshape(torch.linalg.norm(relu3(c1 - c2 + margin
                                                           ), axis=1),[-1,1])

        shapeLimit += torch.reshape(torch.linalg.norm(relu4(d1 - d2 + margin
                                                            ), axis=1),[-1,1])
        shapeLimit += torch.reshape(torch.linalg.norm(relu5(e1 - e2 + margin
                                                            ), axis=1),[-1,1])


        return leftBottomLimit+righttopLimit+shapeLimit+ self.reg(c1) + self.reg(c2) + self.reg(d1)+ self.reg(d2) + self.reg(e1)+ self.reg(e2)
        # return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3) +shapeLimit



    # cClass isSubSet of relation some dClass
    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c1 = c[:, :self.embedding_dim]#.detach()
        d1 = d[:, :self.embedding_dim]#.detach()

        c2 = c[:, self.embedding_dim:]#.detach()
        d2 = d[:, self.embedding_dim:]#.detach()

        transfer_matrix = self.transfer_matrix(input[:, 1]).reshape(-1,self.embedding_dim,self.embedding_dim)
        #
        # c_a = torch.matmul((c2 - c1).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
        #                                                                                           self.embedding_dim)
        # d_a = torch.matmul((d2 - d1).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
        #                                                                                           self.embedding_dim)
        c1 = torch.matmul(c[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
                                                                                                                 self.embedding_dim)
        d1 = torch.matmul(d[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
                                                                                                                 self.embedding_dim)
        # c2 = torch.matmul(c[:, self.embedding_dim:].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
        #                                                                                                  self.embedding_dim)
        # d2 = torch.matmul(d[:, self.embedding_dim:].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
        #                                                                                                          self.embedding_dim)
        #
        c2 = torch.matmul(c[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
                                                                                                                 self.embedding_dim) + torch.matmul(
            (c[:, self.embedding_dim:] - c[:, :self.embedding_dim]).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        d2 = torch.matmul(d[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,
                                                                                                                 self.embedding_dim) + torch.matmul(
            (d[:, self.embedding_dim:] - d[:, :self.embedding_dim]).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)

        rc = torch.linalg.norm(c2 - c1, axis=1) / 2
        rd = torch.linalg.norm(d2 - d1, axis=1) / 2
        x1 = (c1 + c2) / 2
        x2 = (d1 + d2) / 2




        x3 = x1 + r


        euc = torch.linalg.norm(x3 - x2, axis=1)
        relu = torch.nn.ReLU()
        dst = torch.reshape(relu(euc + rc - rd + self.margin), [-1, 1])

        # get new center

        # c1 = c1 + r
        #
        # # get new offset
        # c2 = c2 + r

        # is subset
        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)

        leftBottomLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - c1 -r  + margin, zeros), axis=1),[-1,1])
        righttopLimit =  torch.reshape(torch.linalg.norm(torch.maximum(c2  - d2 +r + margin, zeros), axis=1),[-1,1])
        shapeLimit =  torch.reshape(torch.linalg.norm(torch.maximum(d1 - d2 + margin, zeros), axis=1),[-1,1])

        shapeLimit +=  torch.reshape(torch.linalg.norm(torch.maximum(c1 - c2 + margin, zeros), axis=1),[-1,1])

        return leftBottomLimit + righttopLimit + shapeLimit + self.reg(c1) + self.reg(c2) + self.reg(d1) + self.reg(d2)

        # return dst + self.reg(x1) + self.reg(x2) + shapeLimit
        # cClass is_NOT_SubSet of relation some dClass

    def  neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = c[:, self.embedding_dim:]
        d2 = d[:, self.embedding_dim:]

        transfer_matrix = self.transfer_matrix(input[:, 1]).reshape(-1,self.embedding_dim,self.embedding_dim)

        # c1 = torch.matmul(c1.reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        # d1 = torch.matmul(d1.reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        # c2 = torch.matmul(c2.reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        # d2 = torch.matmul(d2.reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)


        rc = torch.linalg.norm(c2 - c1, axis=1) / 2
        rd = torch.linalg.norm(d2 - d1, axis=1) / 2
        x1 = (c1 + c2) /2
        x2 = (d1 + d2) /2


        x3 = x1 + r


        euc = torch.linalg.norm(x3 - x2, axis=1)

        relu = nn.ReLU()
        dst = torch.reshape((-(euc - rc - rd) + self.margin), [-1, 1])

        startAll = torch.maximum(c1+r, d1)
        endAll = torch.minimum(c2+r, d2)
        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        rightLessLeftLoss = torch.reshape(torch.linalg.norm(torch.maximum(endAll - startAll + margin, zeros), axis=1,ord = 1),
                                          [-1, 1])

        shapeLoss = torch.reshape(torch.linalg.norm(torch.maximum(c1 - c2 + margin,
                                                                  zeros), axis=1), [-1, 1])

        shapeLoss += torch.reshape(torch.linalg.norm(torch.maximum(d1 - d2 + margin,
                                                                   zeros), axis=1), [-1, 1])

        return rightLessLeftLoss  + self.reg(c1) + self.reg(c2) + self.reg(d1)+ self.reg(d2)#+ shapeLoss

        # return dst + self.reg(x1) + self.reg(x2)

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])

        r = self.relationEmbeddingDict(input[:, 0])

        d = self.classEmbeddingDict(input[:, 2])
        c1 = c[:, :self.embedding_dim]#.detach()
        d1 = d[:, :self.embedding_dim]#.detach()

        c2 = c[:, self.embedding_dim:]#.detach()
        d2 = d[:, self.embedding_dim:]#.detach()

        transfer_matrix = self.transfer_matrix(input[:, 0]).reshape(-1,self.embedding_dim,self.embedding_dim)

        # c_a = torch.matmul((c2 - c1).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,self.embedding_dim)
        # d_a = torch.matmul((d2 - d1).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)

        c1 = torch.matmul(c[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        d1 = torch.matmul(d[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        # c2 = torch.matmul(c[:, self.embedding_dim:].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        # d2 = torch.matmul(d[:, self.embedding_dim:].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)
        # #
        # #
        # #
        c2 = torch.matmul(c[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)+torch.matmul((c[:, self.embedding_dim:] - c[:, :self.embedding_dim]).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1,self.embedding_dim)
        d2 = torch.matmul(d[:, :self.embedding_dim].reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)+torch.matmul((d[:, self.embedding_dim:] - d[:, :self.embedding_dim]).reshape(-1, 1, self.embedding_dim), transfer_matrix).reshape(-1, self.embedding_dim)

        rc = torch.reshape(torch.linalg.norm(c2 - c1, axis=1) / 2, [-1, 1])
        rd = torch.reshape(torch.linalg.norm(d2 - d1, axis=1) / 2, [-1, 1])
        x1 = (c1 + c2) / 2
        x2 = (d1 + d2) / 2

        sr = rc + rd


        # c - r should intersect with d
        x3 = x1 - r
        dst = torch.reshape(torch.linalg.norm(x3 - x2, axis=1), [-1, 1])

        relu = torch.nn.ReLU()
        dst_loss = relu(dst - sr - self.margin)
        #return dst_loss + self.reg(x1) + self.reg(x2)
        # get new center
        # c1 = c1 - r
        # # get new offset
        # c2 = c2 - r

        # is subset
        margin = (torch.ones(d1.shape, requires_grad=False) * self.margin).to(self.device)
        zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)

        leftBottomLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - c1+r + margin, zeros), axis=1),[-1,1])
        righttopLimit = torch.reshape(torch.linalg.norm(torch.maximum(c2  - d2-r + margin, zeros), axis=1),[-1,1])

        shapeLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - d2 + margin, zeros), axis=1),[-1,1])

        shapeLimit += torch.reshape(torch.linalg.norm(torch.maximum(c1 - c2 + margin, zeros), axis=1),[-1,1])

        return leftBottomLimit + righttopLimit + shapeLimit+ self.reg(c1) + self.reg(c2) + self.reg(d1)+ self.reg(d2)
        # return dst + self.reg(x1) + self.reg(x2) + shapeLimit

    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = c[:, self.embedding_dim:]
        d2 = d[:, self.embedding_dim:]


        rc = torch.reshape(torch.linalg.norm(c2 - c1, axis=1) / 2, [-1, 1])
        rd = torch.reshape(torch.linalg.norm(d2 - d1, axis=1) /2, [-1, 1])
        x1 = (c1 + c2) / 2
        x2 = (d1 + d2) /2



        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]


        sr = rc + rd


        dst = torch.reshape(torch.linalg.norm(x2 - x1, axis=1), [-1, 1])
        relu = torch.nn.ReLU()

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c2, d2)
        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        zeros = (torch.zeros(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        relu1 = nn.ReLU()
        relu2 = nn.ReLU()
        relu3 = nn.ReLU()

        rightLessLeftLoss = torch.reshape(torch.linalg.norm(relu1(endAll - startAll + margin), axis=1,ord = 1),[-1,1])

        shapeLoss  = torch.reshape(torch.linalg.norm(relu2(c1 - c2 + margin
                                                    ), axis=1),[-1,1])

        shapeLoss += torch.reshape(torch.linalg.norm(relu3(d1 - d2 + margin
                                                    ), axis=1),[-1,1])


        return rightLessLeftLoss + shapeLoss + self.reg(c1) + self.reg(c2) + self.reg(d1)+ self.reg(d2)# +self.reg(d1) +  self.reg(d2)#
        # return dst+self.reg(x1) + self.reg(x2) + shapeLoss

    def top_loss(self, input):
        d = self.classEmbeddingDict(input)
       # print(input)
        d1 = d[:, :self.embedding_dim]

        d2 = d[:, self.embedding_dim:]
        margin = (torch.ones(d1.shape, requires_grad=False) * self.inf).to(self.device)

        zeros = (torch.zeros(d1.shape, requires_grad=False) * self.margin).to(self.device)
        shapeLoss = torch.reshape(torch.linalg.norm(torch.maximum(d1 - d2 + margin,
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

        totalLoss = loss1+ disJointLoss+ topLoss + loss2# + loss3+ loss4 #+negLoss  # + topLoss
        print('loss1=',(loss1).sum().item()/batch,'loss2=',(loss2).sum().item()/batch,'loss3=',(loss3).sum().item()/batch,
              'loss4=',(loss4).sum().item()/batch,'disJointLoss=',disJointLoss.sum().item()/batch,'negLoss=',negLoss.sum().item()/batch,)

        # print(torch.sum(totalLoss*totalLoss))
        # print(torch.sqrt(torch.sum(totalLoss*totalLoss)))
        # print(torch.sum(loss1 * loss1) / batch,torch.sum(loss2 * loss2) / batch,torch.sum(loss3 * loss3) / batch
        #       ,torch.sum(loss4 * loss4) / batch,torch.sum(disJointLoss * disJointLoss) / batch,torch.sum(negLoss * negLoss) / batch)
        return torch.sum(totalLoss*totalLoss) / batch





