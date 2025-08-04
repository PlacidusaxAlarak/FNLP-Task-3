import os

from huggingface_hub import list_repo_refs

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.optim as optim
import time
import math
from torch.cuda.amp import GradScaler, autocast
import config
from model import LanguageModel
from data_utils import load_data_and_prepare, create_dataloaders
from plot_utils import plot_losses, plot_lm_loss_comparison

def get_lr(it):
    #实现Warm up+Consine Decay
    #线性预热
    if it<config.WARMUP_ITERS:
        return config.LEARNING_RATE*it/config.WARMUP_ITERS
    #迭代次数超过衰减周期，减少最小学习率
    if it>config.LR_DECAY_ITERS:
        return config.MIN_LR

    #余弦退火
    decay_ratio=(it-config.WARMUP_ITERS)/(config.LR_DECAY_ITERS-config.WARMUP_ITERS)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))#从1变化成0
    return config.MIN_LR+coeff*(config.LEARNING_RATE-config.MIN_LR)
def run_experiment(exp_config):
    #更新全局配置
    config.TOKENIZER_TYPE=exp_config['tokenizer_type']
    print(f"\n{'='*20} 开始实验: {exp_config['name']} {'='*20}")
    #数据准备
    train_data, val_data, _=load_data_and_prepare(exp_config)
    #创建DataLoader
    train_loader, val_loader=create_dataloaders(train_data, val_data)
    #模型初始化
    model=LanguageModel()
    model.to(config.DEVICE)
    print(f"模型参量:{sum(p.numel() for p in model.parameters())/1e6:2f}M")

    #优化器
    optimizer=optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.95))

    #自动混合精度训练
    scaler=torch.amp.GradScaler(enabled=(config.DEVICE=='cuda'))
    print(f"自动混合精度(AMP)训练:{'已启用'if scaler.is_enabled() else '未启用'}")

    dtype=torch.float16

    #训练村换
    val_losses=[]
    eval_iters_list=[]
    all_lrs=[]
    @torch.no_grad()
    def estimate_loss():
        out={}
        model.eval()
        for split, loader in[('train', train_loader), ('val', val_loader)]:
            losses=[]
            #只评估前十个批次来节省时间
            for i, (X, Y) in enumerate(loader):
                if i>=10:
                    break
                X, Y=X.to(config.DEVICE, non_blocking=True), Y.to(config.DEVICE, non_blocking=True)
                with torch.autocast(device_type=config.DEVICE, dtype=dtype, enabled=(config.DEVICE=='cuda')):
                    _, loss, _=model(X, Y)
                losses.append(loss.item())
            out[split]=torch.tensor(losses).mean().item()
        model.train()
        return out
    print("开始训练")
    start_time=time.time()

    train_iter=iter(train_loader)
    for iter_num in range(config.MAX_ITERS):

        lr=get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        all_lrs.append(lr)
        try:
            xb, yb=next(train_iter)
        except StopIteration:
            train_iter=iter(train_loader)
            xb, yb=next(train_iter)

        xb, yb=xb.to(config.DEVICE, non_blocking=True), yb.to(config.DEVICE, non_blocking=True)

        #训练循环修改
        with torch.autocast(device_type=config.DEVICE, dtype=dtype, enabled=(config.DEVICE=='cuda')):
            logits, loss, _ = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#加入梯度裁剪
        scaler.step(optimizer)
        scaler.update()

        if iter_num%config.EVAL_INTERVAL==0 or iter_num==config.MAX_ITERS-1:
            losses=estimate_loss()
            val_losses.append(losses['val'])
            eval_iters_list.append(iter_num)
            print(f"迭代 {iter_num}/{config.MAX_ITERS} | 训练损失: {losses['train']:.4f} | 验证损失: {losses['val']:.4f} | 耗时: {time.time() - start_time:.2f}s")
            start_time = time.time()

    print("训练完成")
    model_save_path=f"models/lm_{exp_config['name'].replace(' ', '_').replace('(', '').replace(')', '')}.pth"
    save_dir=os.path.dirname(model_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至:{model_save_path}")
    return eval_iters_list, val_losses

def main():
    experiments=[
        {'name':'BPE (vocab=8000)', 'tokenizer_type':'bpe', 'vocab_size':8000},
        {'name': 'BPE (vocab=16000)', 'tokenizer_type': 'bpe', 'vocab_size': 16000},
        {'name':'BPE (vocab=50000)', 'tokenizer_type':'bpe', 'vocab_size':50000}
    ]

    all_results={}
    for exp_conf in experiments:
        iters, losses=run_experiment(exp_conf)
        all_results[exp_conf['name']]=(iters, losses)

    print("\n\n=======所有实验完成，生成最终对比图")
    plot_lm_loss_comparison(all_results)
if __name__ == "__main__":
    main()
