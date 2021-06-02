#Written by LH 
import torch
import os
import torch.nn as nn
import numpy as np
from transtensor import *
'''
训练分为八步：
1) 把整个batch的输入传入encoder 
2) 把decoder的输入设置为特殊的，初始隐状态设置为encoder最后时刻的隐状态 
3) decoder每次处理一个时刻的forward计算 
4) 如果是teacher forcing，把上个时刻的"正确的"词作为当前输入，否则用上一个时刻的输出作为当前时刻的输入 
5) 计算loss 
6) 反向计算梯度 
7) 对梯度进行裁剪 
8) 更新模型(包括encoder和decoder)参数

问题：teacher forcing用到的"当前正确答案"哪里来的
'''
def train(input_variable, lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,
          encoder_optimizer,decoder_optimizer,vatch_size,clip,max_length=MAX_LENGTH):
    #梯度清空
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #设置device到GPU中
    input_variable=input_variable.to(device)
    lengths=lengths.to(device)
    target_variable=target_variable.to(device)
    mask=mask.to(device)
    #初始化变量
    loss=0
    print_lossed=[]
    n_totals=0
    
    #encoder的forward计算
    encoder_outputs,encoder_hidden=encoder(input_variable,lengths)
    
    #decoder的初始输入是SOS，需要构造(1,batch)的输入，表示第一个时刻的batch个输入
    decoder_input=torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input=decoder_input.to(device)
    decoder_hidden=encoder_hidden[:decoder.n_layers]
    
    #确定是否采用teacher forcing
    use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
    
    
    #一次处理一个时刻
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #下一个时刻的输入是当前正确答案
            decoder_input=target_variable[t].view(1,-1)
            #累计loss
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal
    else:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #下一个时刻的输入是当前模型预测概率最高的值
            _,topi=decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input=decoder_input.to(device)
            #累计loss
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
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
    
    return sum(print_lossed)/n_totals;
    

#trainIters进行n_iterations次minibatch的训练


def trainIters(model_name,voc ,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,embedding,
               encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,
               save_every,clip,corpus_name,loadFilename):
    #随机选择n_iteration个batch的数据(pair)
    training_batches=[Batch2Tensor([random.choice(pairs) for _ in range(batch_size)], voc) for _ in range(n_iteration)]
    #初始化
    print('初始化中...')
    start_iteration=1
    print_loss=0
    if loadFilename:
        start_iteration=checkpoint['iteration']+1
    
    #训练
    print('训练中...')
    for iteration in range(start_iteration,n_iteration+1):
        training_batch=training_batched[iteration-1]
        input_variable,lengths,target_vatiable,mask,max_target_len=training_batch
        
        #训练一个batch
        loss=train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    