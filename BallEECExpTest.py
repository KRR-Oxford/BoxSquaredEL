import numpy as np
import pandas as pd
from scipy.stats import rankdata


embeddingFile= './data/ballClassEmbed.pkl'
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

with open(testFile, 'r') as f:

    for line in f:
        if count == sample_num:
            break



        a, b, c = line.split()
        cin = int(a)
        din = int(b)
        ein = int(c)
        c = embeds_list_tail[cin][:dim]
        d = embeds_list_tail[din][:dim]


        rc = np.abs(c[ -1])
        rd = np.abs(d[ -1])

        x1 = c[0:-1]
        x2 = d[0:-1]

        dif = np.linalg.norm(x1-x2)+0.0

        h = (rc*rc-rd*rd+dif*dif)/(2.0*dif)
        newCen = x1+(h/dif)*(x2-x1)
        newR = (rc*rc - h*h)
        res = []
        for i in embeds_list_tail:
            e = i[:dim]
            re = np.abs(e[ -1])
            x3 = e[ 0:-1]


            res.append(np.linalg.norm(newCen - x3) )


        index = rankdata(res)

        rank = index[ein]
        if rank < 500:
            if rank <= 10:
                top10 += 1
                if rank <= 3:
                    top3 += 1
                    if rank <= 1:
                        top1 += 1

            rankAll += rank

            count += 1

print("ave", rankAll / count, ", top1", top1 / count, ", top3", top3 / count, ", top10", top10 / count)