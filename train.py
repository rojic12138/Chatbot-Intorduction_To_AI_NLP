#Written by LH 
import random
import torch
import os
from torch._C import device
import torch.nn as nn
import numpy as np
from define import *
from transtensor import *
from dictionary import *
from model import *
'''
训练分为七步：
1) 把整个batch的输入传入encoder 
2) decoder每次处理一个时刻的forward计算 
3) 如果是teacher forcing，把上个时刻的"正确的"词作为当前输入，否则用上一个时刻的输出作为当前时刻的输入 
4) 计算loss 
5) 反向计算梯度 
6) 对梯度进行裁剪 
7) 更新模型(包括encoder和decoder)参数
'''

def train(opt, 
          input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, 
          ):

    # 加载设置
    device = opt.device
    batch_size = opt.batch_size
    teacher_forcing_ratio = opt.teacher_forcing_ratio
    clip = opt.clip

    # 梯度清空
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #设置device到GPU中
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    #初始化变量
    loss = 0
    print_losses = []
    n_totals = 0
    
    #encoder的forward计算
    encoder_outputs,encoder_hidden=encoder(input_variable,lengths)
    
    #decoder的初始输入是SOS，需要构造(1,batch)的输入，表示第一个时刻的batch个输入
    decoder_input=torch.LongTensor([[SOS_index for _ in range(batch_size)]])
    decoder_input=decoder_input.to(device)
    decoder_hidden=encoder_hidden[:decoder.n_layers] if encoder_hidden is not None else None
    
    #确定是否采用teacher forcing
    use_teacher_forcing=True if random.random() < teacher_forcing_ratio else False
    
    
    #一次处理一个时刻
    output = torch.zeros((max_target_len + 1, batch_size)).long().to(device)
    output[0,:] = decoder_input
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(output,decoder_hidden,encoder_outputs,t)
            #下一个时刻的输入是当前正确答案
            decoder_input=target_variable[t].view(1,-1)
            output[t+1,:] = decoder_input
            #累计loss
            mask_loss,nTotal = maskNLLLoss(opt, decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal
    else:
        for t in range(max_target_len):
            decoder_output,decoder_hidden = decoder(output,decoder_hidden,encoder_outputs,t)
            #下一个时刻的输入是当前模型预测概率最高的值
            _,topi=decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input=decoder_input.to(device)
            output[t+1,:] = decoder_input
            #累计loss
            mask_loss,nTotal = maskNLLLoss(opt, decoder_output,target_variable[t],mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal
    
    #反向传播
    loss.backward()
    
    #对encoder和decoder进行梯度裁剪
    _=torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _=torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)
    
    #更新
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses)/n_totals;
    

#trainIters进行n_iterations次minibatch的训练


def trainIters(opt, 
               voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               save_dir,
               ):

    # 加载设置
    batch_size = opt.batch_size
    n_iteration = opt.n_iteration
    loadfile_name = opt.load_file
    print_every = opt.print_every
    save_every  = opt.save_every
    model_name = opt.model_name
    corpus_name = opt.corpus_name
    encoder_n_layers = opt.encoder_n_layers
    decoder_n_layers = opt.decoder_n_layers
    hidden_size = opt.hidden_size
    clip = opt.clip

    #随机选择n_iteration个batch的数据(pair)
    training_batches=[Batch2Tensor([random.choice(pairs) for _ in range(batch_size)], voc) for _ in range(n_iteration)]
    #初始化
    print('初始化中...')
    start_iteration = 1
    print_loss = 0
    if loadfile_name:
        checkpoint = torch.load(loadfile_name)
        start_iteration = checkpoint['iteration']+1
    
    #训练
    print('训练中...')
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration-1]
        input_variable, target_variable, lengths, max_target_len, mask = training_batch
        
        #训练一个batch
        loss=train( opt,
                    input_variable, lengths, target_variable, mask, max_target_len, 
                    encoder, decoder, embedding, encoder_optimizer, decoder_optimizer)
        print_loss += loss
        
        #打印进度
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("迭代次数: {}; 完成百分比: {:.1f}%; 平均损失: {:.4f}"
			.format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
            
        #保存checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'
                                     .format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    