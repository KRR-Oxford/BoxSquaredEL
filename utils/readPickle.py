import pickle as pl
a = pl.load(open('../data/relationEmbed.pkl','rb'))
b = pl.load(open('../data/rel_embeddings.pkl','rb'))
for ema, emb in zip(a['embeddings'],b['embeddings']):
    if len(ema) != len(emb):
        print(len(ema))

for ema, emb in zip(a['relations'],b['relations']):
    #if ema != emb:
    print(emb)