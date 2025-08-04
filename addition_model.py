import torch
import torch.nn as nn
from torch.nn import functional as F
import addition_config as config

class Block(nn.Module):
    """一个Transformer块，采用Pre-LN（层归一化前置）架构"""
    def __init__(self,d_model,n_heads,dropout):
        super().__init__()
        assert d_model%n_heads==0
        self.ln1=nn.LayerNorm(d_model)
        self.attn=nn.MultiheadAttention(embed_dim=d_model,num_heads=n_heads,dropout=dropout,batch_first=True)
        self.ln2=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model),
            nn.Dropout(dropout),
        )

    def forward(self,x:torch.Tensor,attn_mask:torch.Tensor)->torch.Tensor:
        # 注意力模块，包含残差连接
        attn_output,_=self.attn(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=attn_mask,need_weights=False)
        x=x+attn_output
        # 前馈网络模块，包含残差连接
        x=x+self.ffn(self.ln2(x))
        return x

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        assert config.D_MODEL % config.N_HEADS == 0
        self.token_embedding_table=nn.Embedding(config.VOCAB_SIZE,config.D_MODEL)
        self.position_embedding_table=nn.Embedding(config.BLOCK_SIZE,config.D_MODEL)
        self.dropout=nn.Dropout(config.DROPOUT)
        self.blocks=nn.ModuleList([Block(config.D_MODEL,config.N_HEADS,config.DROPOUT) for _ in range(config.N_LAYERS)])
        self.ln_f=nn.LayerNorm(config.D_MODEL) # 解码器最后的层归一化
        self.lm_head=nn.Linear(config.D_MODEL,config.VOCAB_SIZE,bias=False)
        # 权重绑定
        self.token_embedding_table.weight=self.lm_head.weight
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_square_subsequent_mask(self,sz):
        # 生成一个上三角矩阵，对角线为0，上三角为-inf
        mask=torch.triu(torch.full((sz, sz), float('-inf'), device=config.DEVICE), diagonal=1)
        return mask

    def forward(self,idx,targets=None):
        B,T=idx.shape
        # Token和Position Embedding
        tok_emb=self.token_embedding_table(idx)
        pos=torch.arange(0,T,dtype=torch.long,device=config.DEVICE)
        pos_emb=self.position_embedding_table(pos)
        x=self.dropout(tok_emb + pos_emb)
        # 生成因果注意力掩码
        attn_mask=self._generate_square_subsequent_mask(T)
        # 通过所有Transformer块
        for block in self.blocks:
            x=block(x,attn_mask)
        # 通过最后的层归一化
        x=self.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            # masked_targets=targets.clone()
            # masked_targets[masked_targets==config.PAD_TOKEN_ID]=-100
            # #找到每个序列中'='的位置
            # eq_token_id=tuple(config.VOCAB).index('=')


            # #将'='之前所有位置的target设置为ignore_index
            # for i in range(B):
            #     eq_indices=(idx[i]==eq_token_id).nonzero(as_tuple=True)[0]
            #     if len(eq_indices)>0:
            #         first_eq_pos=eq_indices[0]
            #         masked_targets[i, :first_eq_pos]=-100
            B,T,C=logits.shape
            logits_view=logits.view(B*T,C)
            targets_view=targets.view(B*T)
            loss =F.cross_entropy(logits_view, targets_view, ignore_index=-100)
        return logits,loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens,eos_token_id):
        self.eval()
        for _ in range(max_new_tokens):
            # 裁剪上下文以防止超出位置编码范围
            idx_cond=idx if idx.size(1)<=config.BLOCK_SIZE else idx[:,-config.BLOCK_SIZE:]
            # 前向传播获取logits
            logits,_=self(idx_cond)
            # 只关心最后一个时间步的输出
            logits=logits[:,-1,:]
            # 使用softmax获得概率
            probs=F.softmax(logits,dim=-1)
            # 贪心采样，选择概率最高的token
            idx_next=torch.argmax(probs,dim=-1,keepdim=True)
            # 如果生成了结束符，则停止
            if idx_next.item()==eos_token_id:
                break
            # 将新生成的token拼接到序列中
            idx=torch.cat((idx,idx_next),dim=1)
        self.train()
        return idx