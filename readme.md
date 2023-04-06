# NLP Demo Project
[English](readme_eng.md)

小白入门NLP，从零开始搭建数据处理流程，GPT模块以及训练模块
旨在帮助广大从零开始学习NLP的同学一个尽可能全面的例子，包括数据清洗，预处理等流程。
以下资源可供参考：
- [NLP综述博客](https://www.deeplearning.ai/resources/natural-language-processing/)
- [GPT3结构详解](https://dugas.ch/artificial_curiosity/GPT_architecture.html)
- [NanoGPT](https://github.com/karpathy/nanoGPT/tree/a82b33b525ca9855d705656387698e13eb8e8d4b)
- [tiktoken](https://github.com/openai/tiktoken)
## 代码结构简介
- data_pipeline.py: 数据预处理流程
- dataset.py: 构建pytorch数据集
- models.py: GPT, Attention等网络模组的定义
- train.py: 训练脚本

**需自行下载数据集以及停词表，并放入'''./data'''文件夹内**