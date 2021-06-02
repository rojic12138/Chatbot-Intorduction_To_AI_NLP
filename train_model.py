from train import trainIters
from config import Config
from evaluate import GreedySearchDecoder, evaluateInput
import os
import jieba
from load_model import *
from model import *

"""
训练模式，启动模型进行训练
"""

def Train():
    # 全局设置
    opt = Config()

    # 启动模型
    encoder, decoder, voc, pairs, embedding = initGenModel(opt)

    print("Training mode")
    encoder.train()
    decoder.train()

    # 初始化 optimizer
    print('Building optimizers')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.learning_rate * opt.decoder_learning_ratio)

    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    if opt.load_file:
        checkpoint = torch.load(opt.load_file)
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # 开始训练
    print("Trainging")
    save_dir = os.path.join("data", "save")

    trainIters(opt.model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, opt.encoder_n_layers, opt.decoder_n_layers, save_dir, opt.n_iteration, opt.batch_size,
               opt.print_every, opt.save_every, opt.clip, opt.processed_data_path, opt.load_file)

    print("Evaluation Mode")
    # 设置为eval model
    encoder.eval()
    decoder.eval()

    # 初始化搜索解码器
    searcher = GreedySearchDecoder(encoder, decoder)

    # 结巴分词准备
    init = "".join(list(jieba.cut("Initialize chatting...")))

    # 开始聊天
    evaluateInput(searcher, voc)

if __name__ == '__main__':
    Train()