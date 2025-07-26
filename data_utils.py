import torch
import config

#分词器
class CharTokenizer:
    #字符级分词器
    def __init__(self, corpus):
        self.chars=sorted(list(set(corpus)))
        self.vocab_size=len(self.chars)
        self.stoi={c:i for i,c in enumerate(self.chars)}
        self.itos={i:ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

def get_tokenizer(corpus):
    if config.TOKENIZER_TYPE=='char':
        return CharTokenizer(corpus)

#数据加载和划分
def load_data_and_prepare():
    with open(config.DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        text=f.read()

    tokenizer=get_tokenizer(text)

    #动态更新词表大小
    config.VOCAB_SIZE=tokenizer.vocab_size
    print(f"词表大小:{config.VOCAB_SIZE}")

    data=torch.tensor(tokenizer.encode(text), dtype=torch.long)

    #划分验证集和训练集
    n=int(0.9*len(data))
    train_data=data[:n]
    val_data=data[n:]

    return train_data, val_data, tokenizer

#数据批处理
def get_batch(split, train_data, val_data):
    data=train_data if split=='train' else val_data
    #随机选择batch_size个起始位置
    ix=torch.randint(len(data)-config.BLOCK_SIZE, (config.BATCH_SIZE,))#防止越界, 一次性生成BATCH_SIZE个

    #x是输入序列,y是目标序列(x向右移动一位)
    x=torch.stack([data[i:i+config.BLOCK_SIZE] for i in ix])
    y=torch.stack([data[i+1:i+config.BLOCK_SIZE+1] for i in ix])

    x, y=x.to(config.DEVICE), y.to(config.DEVICE)
    return x, y