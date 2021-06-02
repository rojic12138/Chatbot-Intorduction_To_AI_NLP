import pickle
import os
import jieba
import re
import logging

from numpy import empty, save
# from zh_tool.langconv import Converter
from dictionary import Vocabulary
import random
from define import *
from transtensor import *

"""
数据的预处理
调用Datapreprocess，获取pairs和voc
使用pairs可以创建batch，通过Batch2Tensor可以获取向量
和官方示例的功能一样
"""

# 句子是否已分词
need_jieba = False
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

# 数据预处理
def DataPreprocess(file_name, voc_file = ""):
    print("\nData preprocessing ... ")
    voc = Vocabulary("zh-voc")
    # 如果需要，加载词典
    if load_voc:
        LoadVoc(voc, voc_file)

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

    voc.Trim(10)  # abandon the words with count less than 10
    # 如果pair中的词不再词典里直接过滤
    filtered_pairs = [pair for pair in clean_pairs if IsVocWordInPair(voc, pair)]
    # 存储 voc
    if save_voc:
        SaveVoc(voc, voc_file)
    return voc, filtered_pairs


# 保存转化好的句子对，句子对用\t分隔
def OutputFile(pairs, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(whitespace.join(pair[0]))
            f.write('\t')
            f.write(whitespace.join(pair[1]))
            f.write('\n')


# 测试batch转换
def TestBatchTrans(pairs, file_name):
    # batch size 设置为5，随机选出一个batch
    test_batch_size = 5
    test_batch = [random.choice(pairs) for i in range(test_batch_size)]
    
    # 大概是核心功能
    batches = Batch2Tensor(test_batch, voc)
    input_tensor, output_tensor, lengths, max_output_len, mask= batches
    
    # 看看效果
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("input_tensor:{}".format(input_tensor))
        f.write('\n')
        f.write("length:{}".format(lengths))
        f.write('\n')
        f.write("output_tensor:{}".format(output_tensor))
        f.write('\n')
        f.write("mask:{}".format(mask))
        f.write('\n')
        f.write("max_outpus_len:{}".format(max_output_len))


if __name__ == "__main__":
    raw_name = 'qingyun_processed'
    raw_file_name = 'raw_data/' + raw_name + '.txt'
    processed_file_name = 'raw_data/' + raw_name + '_p.txt'
    voc_file_name = 'raw_data/' + raw_name + '_voc.txt'
    batch_file_name = 'raw_data/' + raw_name + '_batch.txt'
    
    # 数据预处理
    voc, pairs = DataPreprocess(raw_file_name, voc_file_name)

    whitespace = ' '
    print("\nThe dictionary has {} words.".format(voc.words_num))
    print("\n{} pairs selected.".format(len(pairs)))
    
    # 输出pairs
    # OutputFile(pairs, processed_file_name)

    #测试batch转换
    TestBatchTrans(pairs, batch_file_name)

