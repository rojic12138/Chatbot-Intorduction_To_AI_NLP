## 基于pytorch官方教程的聊天机器人
___
### 设置
- 如需对参数进行更改，直接在`config.py`中更改对应参数
### 运行
- 运行前检查`config.py`中的`load_file`，检查加载模型参数的路径
- 直接运行 `main.py` 加载模型，输入句子得到回答
### 训练
- 训练前检查`config.py`中的`load_file`, `voc_file`, `pairs_file`，检查训练所需文件
- 直接运行`train_model.py`开始训练，训练过程中会打印相应信息，训练结束后会启动eval mode启动聊天机器人
___
### 预处理文件  
  - [raw data](https://disk.pku.edu.cn:443/link/67966F87BBA6EDD2CDAEEB28F77040EB)
  - 预处理前后的文本文件位于`raw_data`目录下
    - 后缀`.tsv`的文本文件为经过以第一轮清洗未分词的文件
    - 后缀`jieba.txt`的文本文件为经过第二轮分词后的文件
  - 使用：在`datapreprocess.py`中更改`files`以决定需要处理哪些文件，记得同时修改`main`函数中的`voc`和`pairs`的存储路径 
  - 预处理后的词典和句对文件位于`data`目录下
    - 后缀`voc.pkl`为词典文件，后缀`pairs.pkl`为句对文件，加载模型需要这两个文件，在`config.py`中修改对应路径
### 加载模型文件
- 修改`config.py`中的`load_file`参数为对应路径下的文件
---
### 参考
- [Vaswani, Ashish, et al. "Attention is all you need."Advances in Neural Information Processing Systems. 2017.](https://arxiv.org/pdf/1706.03762.pdf)
- [Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [pt_chatbot](https://github.com/coranholmes/pt_chatbot)
- [Chinese-Chatbot-PyTorch-Implementation](https://github.com/Doragd/Chinese-Chatbot-PyTorch-Implementation)
