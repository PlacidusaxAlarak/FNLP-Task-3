import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.optim as optim
import time


import config
from model import LanguageModel
from data_utils import load_data_and_prepare, get_batch
from plot_utils import plot_losses

def main():
    #数据准备
    train_data, val_data, tokenizer=load_data_and_prepare()

    #模型初始化
    model=LanguageModel()
    model.to(config.DEVICE)
    print(f"模型参量:{sum(p.numel() for p in model.parameters())/1e6:2f}M")

    #优化器
    optimizer=optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    #训练村换
    train_losses=[]
    val_losses=[]
    eval_iters_list=[]
    start_time=time.time()

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        x, y=get_batch('val', train_data, val_data)
        _, loss=model(x, y)
        model.train()
        return loss.item()

    print("开始训练")
    for iter_num in range(config.EPOCHS):
        val_loss=estimate_loss()
        val_losses.append(val_loss)
        eval_iters_list.append(iter_num)


        xb, yb=get_batch('train', train_data, val_data)

        #前向传播的损失计算
        logits, loss=model(xb, yb)
        train_losses.append(loss.item())

        #反向传播和优化
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        current_train_loss=loss.item()
        print(f"迭代 {iter_num}/{config.EPOCHS} | 训练损失: {current_train_loss:.4f} | 验证损失: {val_loss:.4f} | 耗时: {time.time() - start_time:.2f}s")
        start_time = time.time()

    print(f"训练完成！")

    #保存模型
    save_dir=os.path.dirname(config.MODEL_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"模型已保存至:{config.MODEL_SAVE_PATH}")

    if len(train_losses) >= 100:
        # 如果足够长，就进行滑动平均
        print("对训练损失进行平滑处理...")
        smoothed_train_losses = torch.tensor(train_losses).view(-1, 100).mean(1)
        plot_losses(smoothed_train_losses, val_losses, eval_iters_list)
    else:
        # 如果不够长，就直接使用原始数据绘图
        print("迭代次数不足100，绘制原始训练损失曲线。")
        plot_losses(train_losses, val_losses, eval_iters_list)

if __name__ == "__main__":
    main()
