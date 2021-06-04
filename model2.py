#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embeddings(nn.Module):
    """
    Implements embeddings of the words and adds their positional encodings. 
    """
    def __init__(self, vocab_size, d_model, max_len = 50):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1)
        
    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):   # for each position of the word
            for i in range(0, d_model, 2):   # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)   # include the batch size
        return pe
        
    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:, :embedding.size(1)]   # pe will automatically be expanded with the same batch size as encoded_words
        embedding = self.dropout(embedding)
        return embedding



class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads, d_model):
        
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, 512)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, 512)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, 512) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # (batch_size, h, max_len, max_len)
        weights = F.softmax(scores, dim = -1)           # (batch_size, h, max_len, max_len)
        weights = self.dropout(weights)
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, h * d_k)
        context = context.permute(0,2,1,3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        # (batch_size, max_len, h * d_k)
        interacted = self.concat(context)
        return interacted 



class FeedForward(nn.Module):

    def __init__(self, d_model, middle_dim = 2048):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads,dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        #print(embeddings.shape, "A")
        interacted = self.dropout(self.self_multihead(embeddings.transpose(0, 1), embeddings.transpose(0, 1), embeddings.transpose(0, 1), mask).transpose(0, 1))
        #print(interacted.shape, "B")
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
        

class EncoderTransformer(nn.Module):
    
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, heads = 4):
        super(EncoderTransformer, self).__init__()
        self.d_model = hidden_size
        self.embedding = embedding
        self.encoder = nn.ModuleList([EncoderLayer(hidden_size, heads,dropout) for _ in range(n_layers)])
        
    def forward(self, src_words, length):
        src_embeddings = self.embedding(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, None)
        return src_embeddings, None
                
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings, encoded, target_mask):
        query = self.dropout(self.self_multihead(embeddings.transpose(0, 1), embeddings.transpose(0, 1), embeddings.transpose(0, 1), target_mask).transpose(0, 1))
        query = self.layernorm(query + embeddings)
        interacted = self.dropout(self.src_multihead(query.transpose(0, 1), encoded.transpose(0, 1), encoded.transpose(0, 1), None).transpose(0, 1))
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # ע�⣺decoderÿһ��ֻ�ܴ���һ��ʱ�̵����ݣ���Ϊtʱ�̼������˲��ܼ���t+1ʱ�̡�
        # input_step��shape��(1, 64)��64��batch��1�ǵ�ǰ����Ĵ�ID(������һ��ʱ�̵����)
        # ͨ��embedding����(1, 64, 500)��Ȼ�����dropout��shape���䡣
        print(input_step.shape, "E")
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # ��embedded����GRU����forward����
        # �õ�rnn_output��shape��(1, 64, 500)
        # hidden��(2, 64, 500)����Ϊ�������GRU�����Ե�һά��2��
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # ����ע����Ȩ�أ�����ǰ��ķ�����attn_weights��shape��(64, 1, 10)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # encoder_outputs��(10, 64, 500)
        # encoder_outputs.transpose(0, 1)���shape��(64, 10, 500)
        # bmm�������ľ���˷�����һά��batch�����ǿ��԰�attn_weights����64��(1,10)�ľ���
        # ��encoder_outputs.transpose(0, 1)����64��(10, 500)�ľ���
        # ��ôbmm����64��(1, 10)���� x (10, 500)�������յõ�(64, 1, 500)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # ��context������GRU�����ƴ������
        # rnn_output��(1, 64, 500)���(64, 500)
        rnn_output = rnn_output.squeeze(0)
        # context��(64, 1, 500)���(64, 500)
        context = context.squeeze(1)
        # ƴ�ӵõ�(64, 1000)
        concat_input = torch.cat((rnn_output, context), 1)
        # self.concat��һ������(1000, 500)��
        # self.concat(concat_input)�������(64, 500)
        # Ȼ����tanh��������ر��(-1,1)��concat_output��shape��(64, 500)
        concat_output = torch.tanh(self.concat(concat_input))
        # out��(batch_size, �ʵ��Сvoc.num_words)
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

class DecoderTransformer(nn.Module):
    
    def __init__(self, embedding, hidden_size, output_size, device, n_layers = 1, dropout = 0.1, heads = 4):
        super(DecoderTransformer, self).__init__()
        self.embed = embedding
        self.d_model = hidden_size
        self.vocab_size = output_size
        self.decoder = nn.ModuleList([DecoderLayer(hidden_size, heads, dropout) for _ in range(n_layers)])
        self.logit = nn.Linear(hidden_size, self.vocab_size)
        self.device = device
        
    
    def forward(self, input_step_, last_hidden, encoder_outputs, position):
        # target_mask
        tgt_embeddings = self.embed(input_step_.clone().detach())
        max_length = input_step_.shape[0]
        target_mask = torch.zeros((1,1,max_length, max_length)).to(device)
        for i in range(position+1):
            target_mask[:,:,i,:position+1] = 1
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, encoder_outputs, target_mask)
        
        out = F.softmax(self.logit(tgt_embeddings[position]), dim = 1)
        #print(out.shape, out)
        return out, None
        