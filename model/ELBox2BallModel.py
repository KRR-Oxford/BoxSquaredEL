import numpy as np
import torch.nn as nn
import torch
np.random.seed(12)


class RelationModel(nn.Module):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        super(RelationModel, self).__init__()


        self.mlp = nn.Sequential(
            nn.Linear( 2*embedding_dim, 16*embedding_dim),
           nn.LayerNorm(16*embedding_dim),
            nn.Tanh(),
           #  nn.Linear(16 * embedding_dim,  16*embedding_dim),
           # nn.LayerNorm(embedding_dim * 16),
           #  nn.Tanh(),
           #  nn.Linear(16 * embedding_dim, 16 * embedding_dim),
           # nn.LayerNorm(embedding_dim * 16),
           #  nn.Tanh(),
            nn.Linear(16*embedding_dim,  embedding_dim),
           nn.LayerNorm(embedding_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        #input[:,:50]
        return  input[:,:50]#self.mlp(input  )
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
        self.relationNum = relationNum
        self.device = device
        self.reg_norm=1
        self.inf=5

        self.classEmbeddingDict = nn.Embedding(classNum, embedding_dim*2)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight.data,axis=1).reshape(-1,1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        nn.init.uniform_(self.relationEmbeddingDict.weight,a=-1,b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight.data, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim
        self.relationModel = RelationModel(self.embedding_dim)
        self.tailRelationModel =  RelationModel(self.embedding_dim)
        for m in self.relationModel.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        for m in self.tailRelationModel.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)



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

        c2 = c[:, self.embedding_dim:]
        d2 = d[:, self.embedding_dim:]

        # box

        margin = (torch.ones(c1.shape, requires_grad=False) * self.margin).to(self.device)
        relu1 = nn.ReLU()
        leftBottomLimit = torch.reshape(torch.linalg.norm(relu1(d1 - c1 + margin), axis=1), [-1, 1])
        relu2 = nn.ReLU()
        righttopLimit = torch.reshape(torch.linalg.norm(relu2(c1 + c2 - d2 - d1 + margin), axis=1), [-1, 1])

        relu3 = nn.ReLU()
        shapeLimit = torch.reshape(torch.linalg.norm(relu3(- c2+margin
                                                           ), axis=1, ord=1), [-1, 1])
        relu4 = nn.ReLU()
        shapeLimit += torch.reshape(torch.linalg.norm(relu4(- d2+margin
                                                            ), axis=1, ord=1), [-1, 1])
        return leftBottomLimit + righttopLimit + shapeLimit

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
        endAll = torch.minimum(c1 + c2, d1 + d2)
        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)

        leftBottomLimit = torch.reshape(torch.linalg.norm(relu1(e1 - startAll), axis=1), [-1, 1])
        righttopLimit = torch.reshape(torch.linalg.norm(relu2(endAll - e2 - e1), axis=1), [-1, 1])

        shapeLimit = torch.reshape(torch.linalg.norm(relu3(- c2 + margin
                                                           ), axis=1), [-1, 1])

        shapeLimit += torch.reshape(torch.linalg.norm(relu4(- d2 + margin
                                                            ), axis=1), [-1, 1])
        shapeLimit += torch.reshape(torch.linalg.norm(relu5(- e2 + margin
                                                            ), axis=1), [-1, 1])

        return leftBottomLimit + righttopLimit + shapeLimit

    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = c[:, self.embedding_dim:]
        d2 = d[:, self.embedding_dim:]

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c2+c1, d2+d1)
        margin = (torch.ones(endAll.shape, requires_grad=False) * self.margin).to(self.device)


        relu1 = nn.ReLU()
        relu2 = nn.ReLU()
        relu3 = nn.ReLU()

        rightLessLeftLoss = torch.reshape(torch.linalg.norm(relu1(endAll - startAll + margin), axis=1,ord = 1),[-1,1])

        shapeLoss  = torch.reshape(torch.linalg.norm(relu2(  - c2 + margin
                                                    ), axis=1),[-1,1])

        shapeLoss += torch.reshape(torch.linalg.norm(relu3(  - d2 + margin
                                                    ), axis=1),[-1,1])


        return rightLessLeftLoss + shapeLoss

    def nf3Loss(self, input):
        # c = self.classEmbeddingDict(input[:, 0])
        # r = self.relationEmbeddingDict(input[:, 1])
        # d = self.classEmbeddingDict(input[:, 2])
        #
        # c1_o = c[:, :self.embedding_dim]
        # c2_o = c[:, self.embedding_dim:]
        #
        # d1_o = d[:, :self.embedding_dim]
        # d2_o = d[:, self.embedding_dim:]
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
        # d1 = self.tailRelationModel(d1_input)
        # d2 = self.tailRelationModel(d2_input)
        #
        # margin = (torch.ones(d1.shape, requires_grad=False) * self.margin).to(self.device)
        # zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)
        #
        #
        #
        # leftBottomLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - c1 -r  + margin, zeros), axis=1),[-1,1])
        # righttopLimit =  torch.reshape(torch.linalg.norm(torch.maximum(c2+c1 +r - d2-d1  + margin, zeros), axis=1),[-1,1])
        # shapeLimit =  torch.reshape(torch.linalg.norm(torch.maximum( - d2 + margin, zeros), axis=1),[-1,1])
        #
        # shapeLimit +=  torch.reshape(torch.linalg.norm(torch.maximum(  - c2 + margin, zeros), axis=1),[-1,1])
        #
        # return  leftBottomLimit+righttopLimit+shapeLimit
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c1_o = c[:, :self.embedding_dim]
        c2_o = c[:, self.embedding_dim:]

        d1_o = d[:, :self.embedding_dim]
        d2_o = d[:, self.embedding_dim:]

        c1_input = torch.cat((c1_o, r), dim=1)
        c2_input = torch.cat((c2_o, r), dim=1)

        d1_input = torch.cat((d1_o, r), dim=1)
        d2_input = torch.cat((d2_o, r), dim=1)

        c1 = self.relationModel(c1_input)
        c2 = self.relationModel(c2_input)

        d1 = self.tailRelationModel(d1_input)
        d2 = self.tailRelationModel(d2_input)

        rc = torch.linalg.norm(c2, axis=1).reshape([-1, 1]) / 2
        rd = torch.linalg.norm(d2, axis=1).reshape([-1, 1]) / 2

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c2 + c1 + r, d2 + d1)
        relu1 = nn.ReLU()

        rightLessLeftLoss = torch.reshape(
            torch.sum(relu1(endAll - startAll) + (torch.ones(endAll.shape, requires_grad=False) * 0).to(self.device),
                      dim=1), [-1, 1])
        print('rightLessLeftLoss: ', rightLessLeftLoss.sum().item() / 512)

        x3 = c1+r
        euc = torch.linalg.norm(x3 - d1, axis=1).reshape([-1, 1])

        #   relu = torch.nn
        dst = torch.reshape(((euc + rc - rd) + self.margin), [-1, 1])  # + rightLessLeftLoss
        return dst  # + self.reg(x1) + self.reg(x2)

    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c1_o = c[:, :self.embedding_dim]
        c2_o = c[:, self.embedding_dim:]

        d1_o = d[:, :self.embedding_dim]
        d2_o = d[:, self.embedding_dim:]

        c1_input = torch.cat((c1_o, r), dim=1)
        c2_input = torch.cat((c2_o, r), dim=1)

        d1_input = torch.cat((d1_o, r), dim=1)
        d2_input = torch.cat((d2_o, r), dim=1)

        c1 = self.relationModel(c1_input)
        c2 = self.relationModel(c2_input)

        d1 = self.tailRelationModel(d1_input)
        d2 = self.tailRelationModel(d2_input)



        rc = torch.linalg.norm(c2, axis=1).reshape([-1, 1])/2
        rd = torch.linalg.norm(d2, axis=1).reshape([-1, 1])/2

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c2 + c1+r, d2 + d1)
        relu1 = nn.ReLU()

        rightLessLeftLoss = torch.reshape(torch.sum(relu1(endAll - startAll) + (torch.ones(endAll.shape, requires_grad=False)*0   ).to(self.device) , dim=1), [-1, 1])
        print('rightLessLeftLoss: ',rightLessLeftLoss.sum().item() / 512)

        x3 = c1+r
        euc = torch.linalg.norm(x3    - d1   , axis=1).reshape([-1, 1])

        #   relu = torch.nn
        dst = torch.reshape(((euc-rc-rd  ) + self.margin ), [-1, 1])#+ rightLessLeftLoss
        return dst #+ self.reg(x1) + self.reg(x2)


    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        # c = self.classEmbeddingDict(input[:, 1])
        #
        # r = self.relationEmbeddingDict(input[:, 0])
        #
        # d = self.classEmbeddingDict(input[:, 2])
        #
        # c1_o = c[:, :self.embedding_dim]
        # c2_o = c[:, self.embedding_dim:]
        #
        # d1_o = d[:, :self.embedding_dim]
        # d2_o = d[:, self.embedding_dim:]
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
        # d1 = self.tailRelationModel(d1_input)
        # d2 = self.tailRelationModel(d2_input)
        #
        # # is subset
        # margin = (torch.ones(d1.shape, requires_grad=False) * self.margin).to(self.device)
        # zeros = (torch.zeros(d1.shape, requires_grad=False) ).to(self.device)
        #
        # leftBottomLimit = torch.reshape(torch.linalg.norm(torch.maximum(d1 - c1+r + margin, zeros), axis=1),[-1,1])
        # righttopLimit = torch.reshape(torch.linalg.norm(torch.maximum(c2+c1 -r - d2-d1 + margin, zeros), axis=1),[-1,1])
        #
        # shapeLimit = torch.reshape(torch.linalg.norm(torch.maximum(  - d2 + margin, zeros), axis=1),[-1,1])
        #
        # shapeLimit += torch.reshape(torch.linalg.norm(torch.maximum(  - c2 + margin, zeros), axis=1),[-1,1])
        #
        # return leftBottomLimit + righttopLimit + shapeLimit#+ self.reg(c1) + self.reg(c2) + self.reg(d1)+ self.reg(d2)
        c = self.classEmbeddingDict(input[:, 1])
        r = self.relationEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 2])

        c1_o = c[:, :self.embedding_dim]
        c2_o = c[:, self.embedding_dim:]

        d1_o = d[:, :self.embedding_dim]
        d2_o = d[:, self.embedding_dim:]

        c1_input = torch.cat((c1_o, r), dim=1)
        c2_input = torch.cat((c2_o, r), dim=1)

        d1_input = torch.cat((d1_o, r), dim=1)
        d2_input = torch.cat((d2_o, r), dim=1)

        c1 = self.relationModel(c1_input)
        c2 = self.relationModel(c2_input)

        d1 = self.tailRelationModel(d1_input)
        d2 = self.tailRelationModel(d2_input)

        rc = torch.linalg.norm(c2, axis=1).reshape([-1, 1]) / 2
        rd = torch.linalg.norm(d2, axis=1).reshape([-1, 1]) / 2

        startAll = torch.maximum(c1, d1)
        endAll = torch.minimum(c2 + c1 + r, d2 + d1)
        relu1 = nn.ReLU()

        rightLessLeftLoss = torch.reshape(
            torch.sum(relu1(endAll - startAll) + (torch.ones(endAll.shape, requires_grad=False) * 0).to(self.device),
                      dim=1), [-1, 1])
        print('rightLessLeftLoss: ', rightLessLeftLoss.sum().item() / 512)

        x3 = c1-r
        euc = torch.linalg.norm(x3 - d1, axis=1).reshape([-1, 1])

        #   relu = torch.nn
        dst = torch.reshape(((euc + rc - rd) + self.margin), [-1, 1])  # + rightLessLeftLoss
        return dst  # + self.reg(x1) + self.reg(x2)





    def top_loss(self, input):
        d = self.classEmbeddingDict(input[0])
        rd = torch.reshape(torch.abs(d[-1]), [-1, 1])
        return torch.abs(rd - self.inf)



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
        # rand_index = np.random.choice(len(input['top']), size=batch)
        # topData = input['top'][rand_index]
        # topData = topData.to(self.device)
        # topLoss = self.top_loss(topData)

        # negLoss
        rand_index = np.random.choice(len(input['nf3_neg']), size=batch)
        negData = input['nf3_neg'][rand_index]
        negData = negData.to(self.device)
        negLoss = self.neg_loss(negData)
        print('negLoss',negLoss.sum().item() / batch)
        mseloss = nn.MSELoss(reduce=True)
        negLoss = mseloss(negLoss, torch.ones(negLoss.shape, requires_grad=False).to(self.device) *2)

        # print(loss1,loss2,loss3, loss4,disJointLoss,negLoss)
        print(  loss3, loss4,   negLoss)
#  loss1+loss2+disJointLoss+
        totalLoss =  loss1+loss2+disJointLoss+loss3+loss4+ negLoss  # +negLoss #loss4 +disJointLoss+loss1 + loss2 +  negLoss#+ disJointLoss+ topLoss+ loss3 + loss4 +  negLoss

        return (totalLoss)
