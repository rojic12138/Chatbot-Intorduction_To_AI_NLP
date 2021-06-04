#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import torch
from define import *

"""
将输入的句子转换为向量
流程:  外部直接调用Batch2Tensor

       Batch2Tensor会调用InputHandle和outputHandle

       InputHandle将句子转化为带EOS index并补齐PAD index的矩阵，同时求句子长度
       OutputHandle做了同样的事同时求了mask矩阵
"""

# transfer a sentence to a list of word indexes
def IndexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence] + [EOS_index]


# fill then sentence with PAD_token to match the longest length
def ZeroPadding(sen_list, fill_value = PAD_index):
    return list(itertools.zip_longest(*sen_list, fillvalue = fill_value))


# return a mask matrix where PAD_token is 0, 1 otherwise
def BinaryMatrix(sen_list, value = PAD_index):
    mask = []
    for sentence in sen_list:
        sen_mask = []
        for word in sentence:
            if word == value:
                sen_mask.append(0)
            else:
                sen_mask.append(1)
        mask.append(sen_mask)
    return mask


# The sen_list have no EOS token
def InputHandle(sen_list, voc):
    # index first
    index_sen_list = [IndexesFromSentence(voc, sentence) for sentence in sen_list]
    len_sen_list = torch.tensor([len(index_sen) for index_sen in index_sen_list])
    pad_sen_list = ZeroPadding(index_sen_list)
    tensor_sen_list = torch.LongTensor(pad_sen_list)
    return tensor_sen_list, len_sen_list


def OutPutHandle(sen_list, voc):
    index_sen_list = [IndexesFromSentence(voc, sentence) for sentence in sen_list]
    max_len = max([len(index_sen) for index_sen in index_sen_list])
    pad_sen_list = ZeroPadding(index_sen_list)
    mask = BinaryMatrix(pad_sen_list)
    mask = torch.ByteTensor(mask)
    tensor_sen_list = torch.LongTensor(pad_sen_list)
    return tensor_sen_list, mask, max_len


# 输入batch和voc
def Batch2Tensor(batch, voc):
    # sort by the length of first sentence
    batch.sort(key = lambda x: len(x[0]), reverse = True)
    # 填充input batch和output batch
    input_batch, output_batch = [], []
    for pair in batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # 处理batch，得到输入句子向量，输出句子向量，句子长度列表，最长句子长度，mask矩阵
    input_tensor, length_list = InputHandle(input_batch, voc)
    output_tensor, mask, max_len = OutPutHandle(output_batch, voc)
    return input_tensor, output_tensor, length_list, max_len, mask


if __name__ == "__main__":
    pairs = []
    with open("raw_data/testdata_p.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split(' | ')
            pair = []
            for value in values:
                pair.append(value)
            pairs.append(pair)



