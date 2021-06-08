#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import os
from define import *

"""
字典类
将word与index建立一一映射（包括定义的三个token）
"""

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"<PAD>": PAD_index, "<SOS>": SOS_index, "<EOS>": EOS_index, "<UKN>": UKN_index}
        self.index2word = {PAD_index: "<PAD>", SOS_index: "<SOS>", EOS_index: "<EOS>", UKN_index: "<UKN>"}
        self.word2count = {}
        self.words_num = 4  # pad, sos, eos
    
    def InitializeVoc(self, word_dict):
        # 初始化词典
        for word, index in word_dict.items():
            self.word2index[word] = index
            self.index2word[index] = word
            self.word2count[word] = 0
            self.words_num += 1

    def AddWord(self, word):
        # 将词加入词典
        if word in self.word2count.keys():
            self.word2count[word] += 1
        else:
            self.word2count[word] = 1
            self.word2index[word] = self.words_num
            self.index2word[self.words_num] = word
            self.words_num += 1

    def Trim(self, min_cnt):
        # 过滤掉频数小于min_cnt的词并重构词典

        if self.trimmed:
            return
        else:
            self.trimmed = True

        word_list = [word for word, cnt in self.word2count.items() if cnt >= min_cnt]

        self.word2index = {"<PAD>": PAD_index, "<SOS>": SOS_index, "<EOS>": EOS_index, "<UKN>": UKN_index}
        self.index2word = {PAD_index: "<PAD>", SOS_index: "<SOS>", EOS_index: "<EOS>", UKN_index: "<UKN>"}
        self.word2count = {}
        self.words_num = 4  # pad, sos, eos
        # rebuild the voc
        for word in word_list:
            self.AddWord(word)

    def IsWordInVoc(self, word):
        # 判断word是否在词典里

        return word in self.word2count.keys()

    def PrintVoc(self):
        #打印词典

        for word, index in self.word2index.items():
            print("word: {} cnt: {}\n".format(word, index))
