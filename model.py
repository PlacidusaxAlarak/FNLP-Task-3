import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import config
from typing import Optional, Tuple

class Block(nn.Module):
    #使用Pre-LN架构
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model%n_heads==0
        self.n_heads=n_heads
        self.head_dim=d_model//n_heads
        #层归一化
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)

        #注意力层的线性投影
        self.q_proj=nn.Linear(d_model, d_model, bias=False)
        self.k_proj=nn.Linear(d_model, d_model, bias=False)
        self.v_proj=nn.Linear(d_model, d_model, bias=False)
        self.out_proj=nn.Linear(d_model, d_model, bias=False)

        #前馈神经网络
        self.ffn=nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout=nn.Dropout(dropout)
    def forward(self, x:torch.Tensor, past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]], use_cache:bool=False)->Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C=x.shape
        #Pre_LN
        x_norm1=self.ln1(x)
        q=self.q_proj(x_norm1).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)#(B, nh, T, hs)
        k=self.k_proj(x_norm1).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v=self.v_proj(x_norm1).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        #KV缓存
        if use_cache:
            if past_key_value:
                past_key, past_value=past_key_value
                k=torch.cat((past_key, k), dim=-2)
                v=torch.cat((past_value, v), dim=-2)
            if k.size(-2)>config.BLOCK_SIZE:
                k=k[:, :, -config.BLOCK_SIZE:, :]
                v=v[:, :, -config.BLOCK_SIZE:, :]

            present_key_value=(k, v)
        else:
            present_key_value=None
        #Flash Attention
        attn_output=F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=not use_cache)
        attn_output=attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output=self.out_proj(attn_output)

        x=x+attn_output

        #前馈神经网络
        x_norm2=self.ln2(x)
        x=x+self.ffn(x_norm2)

        return x, present_key_value


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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, past_key_values: Optional[list] = None, use_cache: bool = False, pos_offset: int = 0):
        B, T = idx.shape

        # 使用传入的 pos_offset 来计算绝对位置，而不是依赖 past_key_values 的长度
        pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=idx.device)

        # Token and Position Embedding
        tok_emb = self.token_embedding_table(idx)  # (B, T, D_MODEL)
        
        # 使用取模操作来支持超过 BLOCK_SIZE 的位置。这是实现无限长度生成的关键。
        pos_emb = self.position_embedding_table(pos % config.BLOCK_SIZE)  # (T, D_MODEL)

        x = self.dropout(tok_emb + pos_emb)

        new_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                new_key_values.append(new_kv)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view, label_smoothing=0.1)

        return logits, loss, new_key_values
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.7, top_k: int = None, top_p: float = None):
        self.eval()
        past_key_values = None
        
        # 记录 prompt 的长度，用于计算后续 token 的绝对位置
        prompt_len = idx.size(1)

        # 首先处理输入的prompt，位置偏移从 0 开始
        _, _, past_key_values = self(idx, use_cache=True, pos_offset=0)

        current_idx = idx[:, -1:]
        # 使用带索引的循环 `i` 来追踪生成了多少个新 token
        for i in range(max_new_tokens):
            # 计算当前要生成的 token 的绝对位置
            current_pos = prompt_len + i
            
            # 调用 forward 时，传入正确的 pos_offset
            logits, _, past_key_values = self(current_idx, past_key_values=past_key_values, use_cache=True, pos_offset=current_pos)

            logits = logits[:, -1, :]

            # --- [采样逻辑保持不变] ---
            if temperature != 1.0:
                logits /= temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            current_idx = idx_next
            
        self.train()
        return idx