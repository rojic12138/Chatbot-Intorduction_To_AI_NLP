# -*- coding: utf-8 -*-

import numpy as np
import pickle
from dictionary import *
from config import *

words = []
idx = 0
word2idx = {}
vectors = {}
opt = Config()
fastTextEmb = 'raw_data/wiki.zh.vec'
vocFile = opt.voc_file
embedding_dim = opt.embedding_dim
with open(fastTextEmb, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        idx += 1
        vect = np.array(line[1:]).astype(np.float64)
        vectors[word] = vect

with open(vocFile, 'rb') as f:
    voc = pickle.load(f)

print("Finish loading fasttest embedding.")
voc_emb = np.zeros((voc.words_num, embedding_dim))  # 共3444个单词

for k in voc.index2word.keys():

    word = voc.index2word[k]
    if word in vectors.keys():
        voc_emb[k] = vectors[word]
    else:
        voc_emb[k] = np.random.normal(scale=0.6, size=(embedding_dim, ))

# TODO 不使用随机而是计算未知词的均值
pickle.dump(voc_emb, open(f'data/embedding.pkl', 'wb'))