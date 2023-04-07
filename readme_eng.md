# NLP Demo Project
[中文](readme.md)

A minimalist implementation for NLP freshman. 
This repo contains all the necessary modules for modern NLP, such as data cleaning, 
data preprocessing, GPT modules from sratch and so on. Hope this repo provides some insight
for studying NLP especially after the anouncement of ChatGPT and GPT4.
Happy learning!!!

Useful resources：
- [NLP summarization blog](https://www.deeplearning.ai/resources/natural-language-processing/)
- [GPT3 on a napkin](https://dugas.ch/artificial_curiosity/GPT_architecture.html)
- [NanoGPT](https://github.com/karpathy/nanoGPT/tree/a82b33b525ca9855d705656387698e13eb8e8d4b)
- [tiktoken](https://github.com/openai/tiktoken)
- [tricks for small model](https://github.com/BlinkDL/RWKV-LM#how-it-works)
## Code structure
- data_pipeline.py: data preprocessing
- dataset.py: build pytorch dataset
- models.py: definitions for GPT and MultiHeadAttention
- train.py: training script

**Need to download dataset yourself and put it under the folder'''./data'''**

**The data preprocessing steps are specially made for Chinese, and it is different for English**