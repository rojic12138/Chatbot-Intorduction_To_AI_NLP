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
### 组织
- 预处理文件  
  - 预处理前后的文本文件位于`raw_data`目录下
    - 后缀`.tsv`的文本文件为经过以第一轮清洗未分词的文件
    - 后缀`jieba.txt`的文本文件未经过第二轮分词后的文件
  - 使用：在`datapreprocess.py`中更改`files`以决定需要处理哪些文件，记得同时修改`main`函数中的`voc`和`pairs`的存储路径 
  - 预处理后的词典和句对文件位于`data`目录下
    - 后缀`voc.pkl`为词典文件，后缀`pairs.pkl`为句对文件，加载模型需要这两个文件，在`config.py`中修改对应路径
    - `voc.pkl`、`pairs.pkl`只包含了qingyun语料
    - `files_voc.pkl`、`files_voc.pkl`包含了如下语料  
      ```python
      file_list = [
      'raw_data/qingyun.tsv',
      'raw_data/chatterbot.tsv',
      'raw_data/xiaohuangji.tsv',
      'raw_data/ptt.tsv',
      'raw_data/douban_single_turn.tsv'
      ]
      ```
  - 预处理词向量，运行`pre_embed.py`文件，在fasttext中的词直接采用词向量，不在的词随机生成词向量，**embedding_dim指定为300**
    - 如果使用第一个模型，请将`hidden_size`与`embedding_dim`设置为一样，也可以修改`model.py`中对应部分，把需要改成embedding_dim的地方改一下