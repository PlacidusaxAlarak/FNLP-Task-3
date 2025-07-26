import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=config.DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)

        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(position*div_term)
        pe[:, 1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)#注册非参数张量缓存区

    def forward(self, x):
        x=x+self.pe[:, :x.size(1)]
        return self.dropout(x)

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #确保D_MODEL可以被N_HEADS整除
        assert config.D_MODEL%config.N_HEADS==0

        self.token_embedding_table=nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.positional_encoding=PositionalEncoding(config.D_MODEL, config.DROPOUT, config.BLOCK_SIZE)

        #PyTorch标准的TransformerDecoderLayer
        decoder_layer=nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=4*config.D_MODEL,
            dropout=config.DROPOUT,
            batch_first=True
        )

        self.trainformer_decoder=nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.N_LAYERS,
            norm=nn.LayerNorm(config.D_MODEL)
        )

        self.lm_head=nn.Linear(config.D_MODEL, config.VOCAB_SIZE)

        #初始化权重
        self.apply(self._init_weights)#对所有子模块进行_init_weights方法

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_square_subsequent_mask(self, sz):
        #生成一个上三角为-inf, 对角线和下三角为0的mask
        mask=(torch.triu(torch.ones(sz, sz))==1).transpose(0, 1)#transpose使得变成下三角矩阵
        mask=mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))#将-inf替换为0
        return mask.to(config.DEVICE)

    def forward(self, idx, targets=None):
        B, T=idx.shape

        #Token ans Position EMbedding
        tok_emb=self.token_embedding_table(idx)#(B, T, D_MODEL)
        x=self.positional_encoding(tok_emb)#(B, T, D_MODEL)

        #Casual Mask
        tgt_mask=self._generate_square_subsequent_mask(T)

        #Decoder
        output=self.trainformer_decoder(tgt=x, memory=x, tgt_mask=tgt_mask, memory_mask=tgt_mask)

        #Linear Layer
        logits=self.lm_head(output)#(B, T, VOCAB_SIZE)

        #Loss Calculation
        loss=None
        if targets is not None:
            B, T, C=logits.shape
            logits_view=logits.view(B*T, C)
            targets_view=targets.view(B*T)
            loss=F.cross_entropy(logits_view, targets_view)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond=idx[:, -config.BLOCK_SIZE:]

            logits, _=self(idx_cond)#如果实现了__call__, 则可以直接使用self

            logits=logits[:, -1, :]#只取最后一个时间步,形状为[batch_size, vocab_size]

            probs=F.softmax(logits, dim=-1)

            idx_next=torch.multinomial(probs, num_samples=1)#随机性

            idx=torch.cat((idx, idx_next), dim=1)#拼接
        self.train()
        return idx