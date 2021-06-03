#Written by LH 
from torch._C import device
from datapreprocess import *
import torch
import os
import torch.nn as nn
from transtensor import *
from dictionary import *
'''
贪心解码算法 :
    每次选择概率最高的词，作为下一个时刻的输入，直到遇到EOS或者达到最大长度
输入序列(input_seq)，其shape是(input_seq length, 1)， 输入长度input_length和最大输出长度max_length
过程：
1) 把输入传给Encoder，得到所有时刻的输出和最后一个时刻的隐状态。 
2) 把Encoder最后时刻的隐状态作为Decoder的初始状态。 
3) Decoder的第一输入初始化为SOS。 
4) 定义保存解码结果的tensor 
5) 循环直到最大解码长度 
    a) 把当前输入传入Decoder 
    b) 得到概率最大的词以及概率 
    c) 把这个词和概率保存下来 
    d) 把当前输出的词作为下一个时刻的输入 
6) 返回所有的词和概率
#也可以使用Beam-Search算法，也就是每个时刻保留概率最高的Top K个结果，然后下一个时刻尝试把这K个结果输入(当然需要能恢复RNN的状态)，然后再从中选择概率最高的K个。
'''

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(GreedySearchDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.device=device

    def forward(self,input_seq, input_length, max_length):
        device = self.device
        #encoder的forward计算 
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        #将encoder最后时刻的隐状态作为docoder的初始值
        decoder_hidden=encoder_hidden[:self.decoder.n_layers]
        #Decoder的初始输入是SOS
        decoder_input=torch.ones(1,1,device=device,dtype=torch.long)*SOS_index
        #保存解码结果
        all_tokens=torch.zeros([0],device=device,dtype=torch.long)
        all_scores=torch.zeros([0],device=device)
        
        for _ in range(max_length):
            #decoder forward一步
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, 
								encoder_outputs)
            # decoder_outputs是(batch=1, vob_size)
            #使用max返回概率最大的词和得分
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            if decoder_input == EOS_index:
                break
            # 把解码结果保存到all_tokens和all_scores里
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # decoder_input是当前时刻输出的词的ID，这是个一维的向量，因为max会减少一维。
            # 但是decoder要求有一个batch维度，因此用unsqueeze增加batch维度。
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回所有的词和得分。
        return all_tokens, all_scores
    
    
'''
测试从终端输入一个句子，看bot的回复
过程：
1)分词 变成ID
2)传入解码器得到ID 
3)转换成词
'''
def evaluate(opt, encoder, decoder, searcher, voc, sentence, max_length=MAX_SENTENCE_LEN):
    
    device = opt.device
    #句子->ID
    indexes_batch=[IndexesFromSentence(voc,sentence)]
    lengths=torch.tensor([len(indexes)for indexes in indexes_batch])
    #转置 
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    #放到GPU中
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    #用GreedySearchDecoder的实例searcher来解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    #ID->词
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


'''
evaluateInput是chatbot的用户接口
过程：
1)输入一个句子
2)用evaluate生成回 去
3)掉EOS之后的内容
4)重复对话，直到输入q/quit
'''
    
def evaluateInput(opt, encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('请输入> ')
            #输入q/quit，结束程序
            if input_sentence == 'q' or input_sentence == 'quit': break
            #句子归一化
            input_sentence = NormalizeSentence(input_sentence, True)
            #生成答句
            output_words = evaluate(opt, encoder, decoder, searcher, voc, input_sentence)
            #去掉EOS后面的内容
            words = []
            for word in output_words:
                if word == 'EOS':
                    break
                elif word != 'PAD':
                    words.append(word)
            print('尬聊Bot:', ''.join(words))
        #错误处理
        except KeyError:
            print("Error: Encountered unknown word.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    