import pickle as pl
import numpy as np
a = pl.load(open('../data/ballClassEmbed.pkl','rb'))
b = pl.load(open('../data/cls_embeddings.pkl','rb'))
index =0
print     (   np.sum(a['embeddings'][index]*b['embeddings'][index]) / (np.linalg.norm(a['embeddings'][index])*np.linalg.norm(b['embeddings'][index]) ))
print(a['embeddings'][index],b['embeddings'][index])
# for ema, emb in zip(a['embeddings'],b['embeddings']):
#     if len(ema) != len(emb):
#         print(len(ema))
#
# for ema, emb in zip(a['classes'],b['classes']):
#     if ema != emb:
#         print(emb)

