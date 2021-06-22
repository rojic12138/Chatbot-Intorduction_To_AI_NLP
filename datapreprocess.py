#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import os
import jieba
import re
import logging

from numpy import empty, save
from dictionary import Vocabulary
import random
from define import *
from transtensor import *

"""
数据的预处理
使用前请确保句对用'\t'分隔，根据句子是否分词调整need_jieba，需要分词为True
调用Datapreprocess，获取pairs和voc
Datapreprocess只用于单个文件的处理
批处理文件需要使用FileDatapreprocess
一次批处理产生一个voc和pairs的pkl文件，同时存储分好词的文件，如果不需存储请注释相应
的语句
使用pairs可以创建batch，通过Batch2Tensor可以获取向量
和官方示例的功能一样
"""
voc_save_path = 'data/xiaohuangji_voc.pkl'
pairs_save_path = 'data/xiaohuangji_pairs.pkl'
file_list = [
    'raw_data/xiaohuangji.tsv_jieba.txt',
]

# 句子是否已分词
need_jieba = True
# 保存加载词典开关
save_voc = False
load_voc = False
# 分隔符号
split_char = '\t'

jieba.setLogLevel(logging.INFO)
# 过滤非中文字符
regex = re.compile("[^\u4e00-\u9fa5a-zA-Z0-9]")


# 满足句子长度不超过50返回true
def filterPair(pair):
    return len(pair[0]) <= MAX_SENTENCE_LEN and len(pair[1]) <= MAX_SENTENCE_LEN


# 将pair中的词加入词典
def AddWord2Voc(voc, pair):
    for sentence in pair:
        for word in sentence:
            voc.AddWord(word)

# 判断pair中所有的词是否在voc中
def IsVocWordInPair(voc, pair):
    for sentence in pair:
        for word in sentence:
            if not voc.IsWordInVoc(word):
                return False
    return True

# sentence简化并分词
def NormalizeSentence(sentence, is_input = False):
    sentence.encode('utf-8')
    # 如果需要分词，使用结巴分词
    if need_jieba or is_input:
        sentence = jieba.lcut(regex.sub("", sentence))
    # 已经分好词了，直接split
    else:
        sentence = sentence.split(' ')

    return sentence

# 存储词典
def SaveVoc(voc, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word, index in voc.word2index.items():
            f.write(word + " " + str(index) + "\n")
            
# 加载词典
def LoadVoc(voc, load_path):
    word_dict = {}
    with open(load_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            word_dict[line[0]] = int(line[1])
    voc.InitializeVoc(word_dict)

# 对单个文件的数据预处理
def DataPreprocess(file_name, voc = None, voc_file = ""):
    print("\nData preprocessing ... ")
    if voc is None:
        voc = Vocabulary("zh-voc")
    # 如果需要，加载词典
    # if load_voc:
    #     LoadVoc(voc, voc_file)

    clean_pairs = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            pair = line.strip('\n').split(split_char)
            clean_pair = []

            for sentence in pair:
                # 对pair中的句子正则化
                sentence = NormalizeSentence(sentence)
                clean_pair.append(sentence)

            # 如果句子长度超过50直接过滤掉这个pair
            if filterPair(clean_pair):
                # 将pair中的词加入词典
                AddWord2Voc(voc, clean_pair)
                clean_pairs.append(clean_pair)

    # abandon the words with count less than 5
    voc.Trim(5)  
    # 如果pair中的词不再词典里直接过滤
    filtered_pairs = [pair for pair in clean_pairs if IsVocWordInPair(voc, pair)]
    # 存储 voc
    # if save_voc:
    #     SaveVoc(voc, voc_file)
    return voc, filtered_pairs

# 对多个文件的数据预处理
def FileDataPreprocess(file_list):
    print("\nFliles Data preprocessing ... ")
    voc = Vocabulary("file_zh-voc")
    clean_pairs = []
    for file in file_list:
        # 带有jieba后缀不用再分词了
        if 'jieba' in file:
            need_jieba = False
        else:
            need_jieba = True
        total_pairs = []
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                pair = line.strip('\n').split(split_char)
                clean_pair = []

                for sentence in pair:
                    # 对pair中的句子正则化
                    sentence = NormalizeSentence(sentence)
                    clean_pair.append(sentence)
                # 不用过滤直接保存这个pair供以后使用
                total_pairs.append(clean_pair)

                # 如果句子长度超过50直接过滤掉这个pair
                if filterPair(clean_pair):
                    # 将pair中的词加入词典
                    AddWord2Voc(voc, clean_pair)
                    clean_pairs.append(clean_pair)
            
            if need_jieba:
                # 需要分词才重新存储
                OutputFile(total_pairs, file + "_jieba.txt")
            total_pairs.clear()

    # abandon the words with count less than 5
    voc.Trim(5)  
    # 如果pair中的词不再词典里直接过滤
    filtered_pairs = [pair for pair in clean_pairs if IsVocWordInPair(voc, pair)]
    # 存储 voc
    # if save_voc:
    #     SaveVoc(voc, voc_file)
    return voc, filtered_pairs


# 保存转化好的句子对，句子对用\t分隔
def OutputFile(pairs, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(' '.join(pair[0]))
            f.write('\t')
            f.write(' '.join(pair[1]))
            f.write('\n')


# 测试batch转换
def TestBatchTrans(pairs, file_name = ''):
    # batch size 设置为5，随机选出一个batch
    test_batch_size = 5
    test_batch = [random.choice(pairs) for i in range(test_batch_size)]
    
    # 大概是核心功能
    batches = Batch2Tensor(test_batch, voc)
    

if __name__ == "__main__":
    
    # 数据预处理
    voc, pairs = FileDataPreprocess(file_list)

    whitespace = ' '
    print("\nThe dictionary has {} words.".format(voc.words_num))
    print("\n{} pairs selected.".format(len(pairs)))    
    
    # 保存voc和pairs
    with open(voc_save_path,'wb') as f:
        pickle.dump(voc, f)
    print('voc saved')

    with open(pairs_save_path,'wb') as f:
        pickle.dump(pairs, f)
    print('pairs saved')

    # 测试batch转换
    TestBatchTrans(pairs)