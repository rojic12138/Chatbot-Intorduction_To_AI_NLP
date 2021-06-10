import os
from traitlets import config
from config import Config
from load_model import *
from evaluate import *

def Chat():

    print('Load environment')

    # 载入设置
    opt = Config()
    encoder, decoder, voc, pairs, embedding = initGenModel(opt)
    
    # 调整为eval模式
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder, opt.device)
    
    # 处理输入
    evaluateInput(opt, encoder, decoder, searcher, voc)


if __name__ == '__main__':
   Chat()