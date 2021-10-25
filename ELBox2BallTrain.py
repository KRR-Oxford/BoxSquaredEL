#!/usr/bin/env python
import click as ck
import numpy
import torch.optim as optim
from model.ELBox2BallModel import  ELBox2BallModel
from utils.elDataLoader import load_data, load_valid_data
import logging
import torch
logging.basicConfig(level=logging.INFO)
from utils.plot_embeddings import plot_embeddings
import pandas as pd
import  numpy as np
@ck.command()
#family_normalized.owl
#yeast-classes-normalized.owl
@ck.option(
    '--data-file', '-df', default='data/data-train/yeast-classes-normalized.owl',
    help='Normalized ontology file (Normalizer.groovy)')
@ck.option(
    '--valid-data-file', '-vdf', default='data/valid/4932.protein.links.v10.5.txt',
    help='Validation data set')
@ck.option(
    '--out-classes-file', '-ocf', default='data/cls_embeddings.pkl',
    help='Pandas pkl file with class embeddings')
@ck.option(
    '--out-relations-file', '-orf', default='data/rel_embeddings.pkl',
    help='Pandas pkl file with relation embeddings')
@ck.option(
    '--batch-size', '-bs', default=256,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1000,
    help='Training epochs')
@ck.option(
    '--device', '-d', default='gpu:0',
    help='GPU Device ID')
@ck.option(
    '--embedding-size', '-es', default=50,
    help='Embeddings size')
@ck.option(
    '--reg-norm', '-rn', default=1,
    help='Regularization norm')
@ck.option(
    '--margin', '-m', default=-0.1,
    help='Loss margin')
@ck.option(
    '--learning-rate', '-lr', default=0.01,
    help='Learning rate')
@ck.option(
    '--params-array-index', '-pai', default=-1,
    help='Params array index')
@ck.option(
    '--loss-history-file', '-lhf', default='data/loss_history.csv',
    help='Pandas pkl file with loss history')
def main(data_file, valid_data_file, out_classes_file, out_relations_file,
         batch_size, epochs, device, embedding_size, reg_norm, margin,
         learning_rate, params_array_index, loss_history_file):

    device = torch.device('cpu')

    #training procedure
    train_data, classes, relations = load_data(data_file)
    print(len(relations))
    embedding_dim = 50
    model = ELBox2BallModel(device,len(classes), len(relations), embedding_dim=embedding_dim, margin= -0.1 )

    #
    # checkpoint = torch.load('./netPlot.pkl')
    # model.load_state_dict(checkpoint.state_dict())  # 加载网络权重参数

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #optimizer = TheOptimizerClass()
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    model = model.to(device)
    train(model,train_data, optimizer,classes, relations)
    model.eval()
    # cls_file = 'data/classEmbed_TransRBox' + str(epoch + 1) + '.pkl'
    # rel_file = 'data/relationEmbed_TransRBox' + str(epoch + 1) + '.pkl'
    model = model.to('cpu')



    for i,r in zip(range(len(relations)),model.relationEmbeddingDict.weight):

        cls_file = 'data/classEmbedPlot'+str(i)+'.pkl'

        weights = model.classEmbeddingDict.weight
        c1 = torch.cat((weights[:,:embedding_dim], torch.zeros(weights[:,:embedding_dim].shape)+r),dim = 1)
        c2 = torch.cat((weights[:,embedding_dim:], torch.zeros(weights[:,:embedding_dim].shape)+r),dim = 1)

        c1 = weights[:,:embedding_dim]
        c2 = weights[:,embedding_dim:]

       # classEmbedding = model.relationModel
        c1_output  = c1
        c2_output =  c2
        classEmbedding = weights

     #   classEmbedding = model.classEmbeddingDict.weight

        #print(model.classEmbeddingDict.weight.shape)

        df = pd.DataFrame(
            {'classes': list(classes.keys()),
             'embeddings': list(classEmbedding.clone().detach().cpu().numpy())})
        df.to_pickle(cls_file)



    rel_file = 'data/relationEmbedPlot.pkl'
    df = pd.DataFrame(
        {'relations': list(relations.keys()),
         'embeddings': list(model.relationEmbeddingDict.weight.clone().detach().cpu().numpy())})

    df.to_pickle(rel_file)

    # torch.save(model, './netPlot.pkl')




    #store embedding





#ballRelationEmbed

def train(model, data, optimizer, aclasses, relations, num_epochs= 1000 ):
    print(relations)
    model.train()
    for epoch in range(num_epochs):
        #model.zero_grad()
        loss = model(data)
        print('epoch:',epoch,'loss:',round(loss.item(),3))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch%100==0:
        #
        #     cls_df = list(model.classEmbeddingDict.weight.clone().detach().cpu().numpy())
        #     nb_classes = len(cls_df)
        #
        #     embeds_list = cls_df
        #     classes = {k: v for k, v in enumerate(aclasses)}
        #
        #     size = len(embeds_list[0])
        #     embeds = np.zeros((nb_classes, size), dtype=np.float32)
        #     for i, emb in enumerate(embeds_list):
        #         embeds[i, :] = emb
        #     l1 = embeds[:, :-2]
        #     r1 = embeds[:, 2:]
        #     plot_embeddings(l1,l1+ r1,  classes, epoch)



     #    for key in classes.keys():
     #        currentClass = torch.tensor(classes[key])
     #        embedding = torch.tensor(model.classEmbeddingDict(currentClass))
     #        # classCenter = model.centerTransModel(embedding)
     #        # classOffset = model.offsetTransModel(embedding)
     #       # print(key, embedding+torch.Tensor([2]))
     # #   if (epoch+1)% 2000 == 0:



if __name__ == '__main__':
    main()
