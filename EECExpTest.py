import numpy as np
import pandas as pd
from scipy.stats import rankdata

ifTestBox = False

embeddingFile= './data/classEECEmbed.pkl'
testFile = './data/data-train/interGo.txt'
cls_df_tail = pd.read_pickle(embeddingFile)
embeds_list_tail = cls_df_tail['embeddings'].values
dim=50
sample_num = 100
rankAll= 0
count = 0
top1 =0
top3 = 0
top10 = 0

with open(testFile,'r') as f:
    for line in f:
        if count ==sample_num:
            break
        a, b, c = line.split()
        cin = int(a)
        din = int(b)
        ein = int(c)
        c1= embeds_list_tail[cin][:dim]
        d1=embeds_list_tail[din][:dim]

        c2=np.abs(embeds_list_tail[cin][dim:])
        d2=np.abs(embeds_list_tail[din][dim:])

        startAll = np.maximum(c1-c2, d1-d2)
        endAll = np.minimum(c1 + c2, d1 + d2)
        #

        newR = np.abs(startAll-endAll)/2

        res = []
        for i in embeds_list_tail:

             res.append(np.linalg.norm(( startAll+endAll)/2-i[:dim]))
        index = rankdata(res )

        rank = index[ein]
        if rank<500:
            if rank<=10:
                top10+=1
                if rank<=3:
                    top3+=1
                    if rank<=1:
                        top1+=1

            rankAll+=rank

            count += 1

print("ave",rankAll/count,", top1",top1/count,", top3",top3/count,", top10",top10/count)
