import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.optim as optim
import time


import config
from model import LanguageModel
from data_utils import load_data_and_prepare, get_batch
from plot_utils import plot_losses, plot_lm_loss_comparison


def run_experiment(exp_config):
    #更新全局配置
    config.TOKENIZER_TYPE=exp_config['tokenizer_type']
    if 'vocab_size' in exp_config:
        config.VOCAB_SIZE=exp_config['vocab_size']
    print(f"\n{'='*20} 开始实验: {exp_config['name']} {'='*20}")
    #数据准备
    train_data, val_data, _=load_data_and_prepare()

    #模型初始化
    model=LanguageModel()
    model.to(config.DEVICE)
    print(f"模型参量:{sum(p.numel() for p in model.parameters())/1e6:2f}M")

    #优化器
    optimizer=optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    #训练村换
    val_losses=[]
    eval_iters_list=[]

    @torch.no_grad()
    def estimate_loss():
        out={}
        model.eval()
        for split in ['train', 'val']:
            losses=torch.zeros(10)
            for k in range(10):
                X, Y=get_batch(split, train_data, val_data)
                _, loss=model(X, Y)
                losses[k]=loss.item()
            out[split]=losses.mean()
        model.train()
        return out

    print("开始训练")
    start_time=time.time()
    for iter_num in range(config.EPOCHS):
        xb, yb=get_batch('train', train_data, val_data)
        _, loss=model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num%config.EVAL_INTERVAL==0 or iter_num==config.EPOCHS-1:
            losses=estimate_loss()
            val_losses.append(losses['val'])
            eval_iters_list.append(iter_num)
            print(f"迭代 {iter_num}/{config.EPOCHS} | 训练损失: {losses['train']:.4f} | 验证损失: {losses['val']:.4f} | 耗时: {time.time() - start_time:.2f}s")
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
        {'name':'Char Tokenizer', 'tokenizer_type':'char'},
        {'name':'BPE (vocab=5000)', 'tokenizer_type':'bpe', 'vocab_size':5000}
    ]

    all_results={}
    for exp_conf in experiments:
        iters, losses=run_experiment(exp_conf)
        all_results[exp_conf['name']]=(iters, losses)

    print("\n\n=======所有实验完成，生成最终对比图")
    plot_lm_loss_comparison(all_results)
if __name__ == "__main__":
    main()
