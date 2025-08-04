import torch
from transformers import AutoTokenizer

import config
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import os
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
#分词器

class BPETokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer=Tokenizer.from_file(tokenizer_path)
        config.VOCAB_SIZE=self.tokenizer.get_vocab_size()

    def encode(self, s):
        return self.tokenizer.encode(s).ids

    def decode(self, s):
        return self.tokenizer.decode(s)


# 新增：用于加载预训练分词器的包装类
class PretrainedTokenizer:
    def __init__(self, tokenizer_name: str):
        # 这一行是关键：使用正确的方式加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

        # 这一行很重要：自动更新全局配置，让模型知道词表有多大
        print(f"加载预训练分词器 '{tokenizer_name}'，词表大小为: {self.tokenizer.vocab_size}")
        config.VOCAB_SIZE = self.tokenizer.vocab_size

    def encode(self, s: str):
        return self.tokenizer.encode(s)

    def decode(self, ids: list[int]):
        return self.tokenizer.decode(ids)

def get_tokenizer_for_generation(exp_config):
    tokenizer_type=exp_config['tokenizer_type']
    corpus_text=None
    tokenizer=get_tokenizer(
        tokenizer_type,
        corpus=corpus_text,
        vocab_size=exp_config.get('vocab_size'),
        tokenizer_name=exp_config.get('tokenizer_name'),
        #强制离线加载
        local_files_only=True
    )
    print(f"分词器类型: {config.TOKENIZER_TYPE} 词表大小: {config.VOCAB_SIZE}")
    return tokenizer
def train_bpe_tokenizer(corpus_path, vocab_size):
    #训练一个新的BPE分词器并保存
    tokenizer_path=f"tokenizers/bpe_vocab{vocab_size}.json"
    if os.path.exists(tokenizer_path):
        print(f"分词器已存在:{tokenizer_path}")
        return tokenizer_path

    print(f"正在训练BPE分词器，词表大小:{vocab_size}...")
    tokenizer=Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer=Whitespace()
    trainer=BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

    tokenizer.train([corpus_path], trainer)

    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"分词器已保存至:{tokenizer_path}")
    return tokenizer_path
def get_tokenizer(tokenizer_type, corpus=None, vocab_size=None, tokenizer_name=None, local_files_only=True):
    if tokenizer_type=='char':
        return CharTokenizer(corpus)
    elif config.TOKENIZER_TYPE=='bpe':
        tokenizer_path=train_bpe_tokenizer(config.DATA_FILE_PATH, vocab_size)
        return BPETokenizer(tokenizer_path)
    elif tokenizer_type=='pretrained':
        assert tokenizer_name is not None
        return PretrainedTokenizer(tokenizer_name)
    else:
        raise ValueError(f"未知的分词器类型:{tokenizer_type}")

#数据加载和划分
def load_data_and_prepare(exp_config):
    tokenizer_type=exp_config['tokenizer_type']
    with open(config.DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        text=f.read()

    tokenizer=get_tokenizer(
        tokenizer_type,
        corpus=text if tokenizer_type=='char' else None,
        vocab_size=exp_config.get('vocab_size'),
        tokenizer_name=exp_config.get('tokenizer_name')
    )

    #动态更新词表大小

    print(f"分词器类型:{config.TOKENIZER_TYPE} 词表大小:{config.VOCAB_SIZE}")

    data=torch.tensor(tokenizer.encode(text), dtype=torch.long)

    #划分验证集和训练集
    n=int(0.9*len(data))
    train_data=data[:n]
    val_data=data[n:]

    return train_data, val_data, tokenizer

class LanguageModelDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        super().__init__()
        self.data=data_tensor
        self.block_size=block_size

    def __len__(self):
        return len(self.data)-self.block_size

    def __getitem__(self, idx):
        chunk=self.data[idx:idx+self.block_size+1]
        #x是前block_size个token
        x=chunk[:-1]
        #y是后block_size个token
        y=chunk[1:]
        return x, y

def create_dataloaders(train_data, val_data):
    train_dataset=LanguageModelDataset(train_data, config.BLOCK_SIZE)
    val_dataset=LanguageModelDataset(val_data, config.BLOCK_SIZE)

    num_workers=4 if config.DEVICE=='cuda' else 0
    print(f"使用{num_workers}个进程进行数据加载...")
    train_loader=DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,#加速数据从CPU到GPU的传输
        drop_last=True,#丢弃最后一个不完整的batch
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
#数据批处理
