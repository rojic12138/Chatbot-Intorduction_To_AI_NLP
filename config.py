import torch

class Config:
    model_name = 'mole'
    '''
    Chatbot模型参数
    '''
    corpus_name = 'qingyun'                       # 数据来源
    voc_file = 'data/voc.pkl'                     # 词典文件
    pairs_file = 'data/pairs.pkl'                 # 对话数据文件
    load_file = None
    embedding_file = None                         # 词向量文件，暂时没有
    max_input_length = 50                         #输入的最大句子长度
    max_generate_length = 20                      #生成的最大句子长度

    '''
    训练超参数
    '''
    batch_size = 32
    shuffle = True                                #dataloader是否打乱数据
    num_workers = 0                               #dataloader多进程提取数据
    bidirectional = True                          #Encoder-RNN是否双向
    hidden_size = 240
    embedding_dim = 256
    attn_method = 'dot'                           #attention method
    dropout = 0                                   #是否使用dropout
    clip = 50.0                                   #梯度裁剪阈值
    encoder_n_layers = 2                          #Encoder-RNN层数
    decoder_n_layers = 2                          #decoder层数                              
    learning_rate = 1e-5
    teacher_forcing_ratio = 1.0                   #teacher_forcing比例
    decoder_learning_ratio = 5.0
    '''
    训练周期信息
    '''
    n_iteration = 100000                            # 训练次数
    print_every = 10
    save_every = 500                              # 存储训练的数据
    '''
    GPU
    '''
    use_gpu = torch.cuda.is_available() #是否使用gpu
    device = torch.device("cuda" if use_gpu else "cpu") #device
