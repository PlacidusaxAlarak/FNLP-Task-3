import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import config



class Block(nn.Module):
    #使用Pre-LN架构
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model%n_heads==0

        #层归一化
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)

        #多头自注意力机制
        self.attn=nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        #前馈神经网络
        self.ffn=nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x:torch.Tensor, attn_mask:torch.Tensor)->torch.Tensor:
        #Pre_LN
        x_norm1=self.ln1(x)
        attn_output, _=self.attn(x_norm1, x_norm1, x_norm1, attn_mask=attn_mask, is_causal=False)
        x=x+attn_output

        #前馈神经网络
        x_norm2=self.ln2(x)
        x=x+self.ffn(x_norm2)

        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #确保D_MODEL可以被N_HEADS整除
        assert config.D_MODEL%config.N_HEADS==0

        self.token_embedding_table=nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.position_embedding_table=nn.Embedding(config.BLOCK_SIZE, config.D_MODEL)
        self.dropout=nn.Dropout(config.DROPOUT)
        self.blocks=nn.ModuleList(
            [Block(config.D_MODEL, config.N_HEADS, config.DROPOUT) for _ in range(config.N_LAYERS)]
        )

        #解码器最后的层归一化
        self.ln_f=nn.LayerNorm(config.D_MODEL)

        #最终的线性层(语言模型头)
        self.lm_head=nn.Linear(config.D_MODEL, config.VOCAB_SIZE, bias=False)

        #权重绑定
        self.token_embedding_table.weight=self.lm_head.weight#两者在空间上高度一致


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
        assert T<=config.BLOCK_SIZE, f"输入序列长度({T})超过了BLOCK_SIZE({config.BLOCK_SIZE})"
        #Token ans Position EMbedding

        tok_emb=self.token_embedding_table(idx)#(B, T, D_MODEL)

        pos=torch.arange(0, T, dtype=torch.long, device=config.DEVICE)
        pos_emb=self.position_embedding_table(pos)#(T, D_MODEL)

        x=self.dropout(tok_emb+pos_emb)

        #Casual Mask
        attn_mask=self._generate_square_subsequent_mask(T)

        for block in self.blocks:
            x=block(x, attn_mask)

        #最终的LN层
        x=self.ln_f(x)

        logits=self.lm_head(x)

        loss=None
        if targets is not None:
            B, T, C=logits.shape
            logits_view=logits.view(B*T, C)
            targets_view=targets.view(B*T)
            loss=F.cross_entropy(logits_view, targets_view, label_smoothing=0.1)

        return logits, loss
    def generate(self, idx:torch.Tensor, max_new_tokens:int, temperature:float=0.7, top_k:int=None, top_p:float=None):
       self.eval()
       for _ in range(max_new_tokens):
           #输入裁剪到BLOCK_SIZE
            idx_cond=idx[:, -config.BLOCK_SIZE:]

            logits, _=self(idx_cond)
            logits =logits[:, -1, :]

            if temperature!=1.0:
                logits/=temperature

            if top_k is not None:
                v, _=torch.topk(logits, min(top_k, logits.size(-1)))
                #所有低于低k个单词概率的logits都置为负无穷
                logits[logits<v[:, [-1]]]=-float('-inf')

            probs=F.softmax(logits, dim=-1)

            if top_p is not None:
               sorted_probs, sorted_indices=torch.sort(probs, descending=True)
               cumulative_probs=torch.cumsum(sorted_probs, dim=-1)#计算累计概率

               sorted_indices_to_remove=cumulative_probs>top_p#找到第一个累计概率超过p的位置，并且标记此后所有的词为"待移除"
               #至少留一个词
               sorted_indices_to_remove[..., 1:]=sorted_indices_to_remove[..., :-1].clone()
               sorted_indices_to_remove[..., 0]=0

               #根据索引找到要移除的词，并将它们的概率设为0
               indices_to_remove=torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
               probs[indices_to_remove]=0

               #重新归一化
               probs=probs/probs.sum(dim=-1, keepdim=True)

            idx_next=torch.multinomial(probs, num_samples=1)#随机抽取, idx_next就是新生成的新词

            idx=torch.cat((idx, idx_next), dim=1)

       self.train()
       return idx