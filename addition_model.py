import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import addition_config as config

class PositionalEncoding(nn.Module):
    #为输入序列添加位置信息
    def __init__(self, d_model, dropout=config.DROPOUT, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(position*div_term)#编入位置信息
        pe[:, 1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x=x+self.pe[:, :x.size(1)]
        return self.dropout(x)

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        assert config.D_MODEL % config.N_HEADS ==0

        self.token_embedding_table=nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.positional_encoding=PositionalEncoding(config.D_MODEL, config.DROPOUT, config.BLOCK_SIZE)

        decoder_layer=nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.D_MODEL*4,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_decoder=nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.N_LAYERS,
            norm=nn.LayerNorm(config.D_MODEL)
        )

        self.lm_head=nn.Linear(config.D_MODEL, config.VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):#初始化权重
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_square_subsequent_mask(self, sz):#生成掩码
        mask=torch.triu(torch.ones(sz, sz)==1).transpose(0, 1)
        mask=mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask.to(config.DEVICE)

    def forward(self, idx, targets=None):
        B, T=idx.shape
        tok_emb=self.token_embedding_table(idx)#整数ID替换成(BATCH_SIZE, T, D_MODEL)
        x=self.positional_encoding(tok_emb*math.sqrt(config.D_MODEL))#防止梯度过小
        tgt_mask=self._generate_square_subsequent_mask(T)#(T, T)位置掩码
        output=self.transformer_decoder(tgt=x, memory=x, tgt_mask=tgt_mask, memory_mask=tgt_mask)
        logits=self.lm_head(output)

        loss=None
        if targets is not None:
            B, T, C=logits.shape
            logits_view=logits.view(B*T, C)
            targets_view=targets.view(B*T)
            loss=F.cross_entropy(logits_view, targets_view)

        return logits, loss

    @torch.no_grad()#整个函数的所有内容都是无梯度操作
    def generate(self, idx, max_new_tokens, eos_token_id):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond=idx[:, -config.BLOCK_SIZE:]#上下文窗口，末尾向前取BLOCK_SIZE个token
            logits, _=self(idx_cond)#推理时loss是None，使用_忽略
            logits=logits[:, -1, :]#只取最后一个时间步的输出，
            probs=F.softmax(logits, dim=-1)
            idx_next=torch.multinomial(probs, num_samples=1)
            if idx_next.item()==eos_token_id:#计算出该终止了
                break
            idx=torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
