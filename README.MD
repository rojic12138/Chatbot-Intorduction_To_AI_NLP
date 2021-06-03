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
**施工中**