import addition_config as config
from plot_utils import plot_losses
import random
import torch

class MathTokenizer:
    def __init__(self, vocab):
        self.vocab=vocab
        self.vocab_size=len(vocab)
        self.stoi={ch:i for i, ch in enumerate(vocab)}
        self.itos={i:ch for i, ch in enumerate(vocab)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


def generate_addition_problem(num_digits1, num_digits2, max_digits=4):
    n1=random.randint(0, 10**num_digits1-1)
    n2=random.randint(0, 10**num_digits2-1)
    sum_val=n1+n2
    sum_rev=str(sum_val)[::-1]
    #使用PAD_TOKEN进行左对齐填充
    s1_padded=str(n1).rjust(max_digits, config.PAD_TOKEN)
    s2_padded=str(n2).rjust(max_digits, config.PAD_TOKEN)
    sum_rev_padded=sum_rev.ljust(max_digits + 1, config.PAD_TOKEN)
    problem=f"{s1_padded}+{s2_padded}={sum_rev_padded}\n"
    return problem

def generate_dataset(num_samples, allowed_digit_pairs):
    data=[]
    for _ in range(num_samples):
        d1, d2=random.choice(allowed_digit_pairs)
        data.append(generate_addition_problem(d1, d2))
    return "".join(data)

def get_batch(split, train_data, val_data):
    """从数据中获取一个批次，用于训练或验证。"""
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x=torch.stack([data[i:i+config.BLOCK_SIZE] for i in ix])#拼接张量，使得每个张量的形状都是(BATCH_SIZE, BLOCK_SIZE)
    y=torch.stack([data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    return x, y