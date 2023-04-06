'''
Data preprocess for NLP
'''
import re
from transformers import BertTokenizer

class DataPipeline():
    
    def __init__(self, file_name, stop_words_file) -> None:
        with open(file_name, 'r') as f:
            self.data = f.read()
        self.stop_words = set(open(stop_words_file, 'r', encoding='utf-8').read().splitlines())
        self.tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-couplet")

    def _normalize(self, contents):
        '''
        https://zhuanlan.zhihu.com/p/407918235
        '''
        contents = contents.split('\n')
        def _per_sentence(content):
            content = re.sub(r"^第.+?卷", "", content)
            content = re.sub(r"^第.+?回", "", content)
            content = re.sub(r'[^\u4e00-\u9fff\s\u2000-\u206F\u2E00-\u2E7F\\，。！？]+', '', content)

            return content
        contents = list(map(_per_sentence, contents))
        # convert list of sentences to a giant string
        contents = ''.join(content for content in contents if content is not None)
        return contents
    
    def _remove_stop_words(self, contents):
        contents = "".join(word for word in contents if word not in self.stop_words)
        return contents
    
    def _tokenize(self, contents):
        # max_len = self.tokenizer.model_max_len
        contents = self.tokenizer(contents, return_tensors='pt')
        return contents

    def process(self):
        # 移除数据中没有用的token
        contents = self._normalize(self.data)

        # 移除停词
        contents = self._remove_stop_words(contents)

        # Tokenize
        contents = self._tokenize(contents)
        vocab_size = self.tokenizer.vocab_size
        return contents, vocab_size


if __name__ == "__main__":
    old_data = './data/red_UTF82.txt'
    stop_words = './data/stop_words.txt'
    pipeline = DataPipeline(old_data, stop_words)
    contents, vocab_size = pipeline.process()
    print(contents)
    





