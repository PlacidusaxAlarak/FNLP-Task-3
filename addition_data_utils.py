import addition_config as config
from plot_utils import plot_losses
import random
import torch
from torch.utils.data import Dataset

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


def generate_addition_problem(num_digits1, num_digits2, max_digits=config.MAX_DIGITS):
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

class AdditionDataset(Dataset):
    def __init__(self, data_str:str, tokenizer:MathTokenizer, block_size:int):
        self.block_size=block_size
        self.data=torch.tensor(tokenizer.encode(data_str), dtype=torch.long)

    def __len__(self):
        return len(self.data)-self.block_size

    def __getitem__(self, idx):
        x=self.data[idx:idx+self.block_size]
        y=self.data[idx+1:idx+self.block_size+1]
        return x, y