import torch
import os
import torch.nn as nn
import numpy as np
from config import Config
from transtensor import *
from dictionary import *
from model import *


def initGenModel(opt):
    # 加载词典
    with open(opt.voc_file, 'rb') as f:
        voc = pickle.load(f)
    # 加载pairs
    with open(opt.pairs_file, 'rb') as f:
        pairs = pickle.load(f)

    # 如果有load file 进行加载
    if opt.load_file:
        checkpoint = torch.load(opt.load_file)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    
    # 初始化词向量
    print('Initialize word embeddings')
    embedding = nn.Embedding(voc.words_num, opt.hidden_size)

    # 载入预训练的词向量
    if opt.load_file:
        embedding.load_state_dict(embedding_sd)
    elif opt.embedding_file:
        with open(opt.embedding_file, 'rb') as f:
            emb = pickle.load(f)
        emb = torch.from_numpy(emb)
        embedding.load_state_dict({'weight': emb})

    # 初始化模型
    print('Initilize model')
    #encoder = EncoderRNN(opt.hidden_size, embedding, opt.encoder_n_layers, opt.dropout)
    encoder = EncoderTransformer(opt.hidden_size, embedding, opt.encoder_n_layers, opt.dropout)
    #decoder = LuongAttnDecoderRNN(opt.attn_method, embedding, opt.hidden_size, voc.words_num, opt.decoder_n_layers, opt.dropout)
    decoder = DecoderTransformer(embedding, opt.hidden_size, voc.words_num, opt.device, opt.decoder_n_layers, opt.dropout, heads = 4)
    if opt.load_file:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        
    encoder = encoder.to(opt.device)
    decoder = decoder.to(opt.device)

    print('Model is ready')
    return encoder, decoder, voc, pairs, embedding
