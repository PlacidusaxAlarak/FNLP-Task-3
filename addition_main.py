import torch
import torch.optim as optim
import random
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import collections
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from torch.optim.lr_scheduler import CosineAnnealingLR
import addition_config as config
from torch.utils.data import DataLoader, random_split
from addition_data_utils import MathTokenizer, generate_dataset, AdditionDataset
from addition_model import AdditionTransformer
from plot_utils import plot_losses
from plot_utils import plot_losses, plot_addition_accuracy_comparison
def run_training(train_loader, val_loader, experiment_name):
    print(f"\n========开始实验:{experiment_name}====")
    model=AdditionTransformer()
    model.to(config.DEVICE)
    print(f"模型参数量:{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer=optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #定义warmup步数
    warmup_steps=int(config.TRAIN_STEPS*0.1)
    warmup_scheduler=LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler=CosineAnnealingLR(optimizer, T_max=config.TRAIN_STEPS-warmup_steps, eta_min=config.LEARNING_RATE/10)
    #余弦退火调度器
    scheduler=SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    train_losses, val_losses, eval_iters_list=[], [], []

    print("开始训练")
    start_time=time.time()
    train_iter=iter(train_loader)
    for step in range(config.TRAIN_STEPS):
        try:
            xb, yb=next(train_iter)
        except StopIteration:
            train_iter=iter(train_loader)
            xb, yb=next(train_iter)
        xb, yb=xb.to(config.DEVICE), yb.to(config.DEVICE)
        logits, loss=model(xb, yb)

        optimizer.zero_grad(set_to_none=True)#优化，不会重新分配内存
        loss.backward()#计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()#更新权重
        scheduler.step()
        if step%config.EVAL_INTERVAL==0 or step==config.TRAIN_STEPS-1:
            model.eval()
            val_loss_avg=0
            with torch.no_grad():
                for val_xb, val_yb in val_loader:
                    val_xb, val_yb=val_xb.to(config.DEVICE), val_yb.to(config.DEVICE)
                    _, val_loss=model(val_xb, val_yb)
                    val_loss_avg+=val_loss.item()
            model.train()
            val_loss_avg/=len(val_loader)
            train_losses.append(loss.item())
            val_losses.append(val_loss_avg)
            eval_iters_list.append(step)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"实验{experiment_name} 步数{step}/{config.TRAIN_STEPS}|训练损失:{loss.item():.4f}|验证损失:{val_loss_avg:.4f}|学习率:{current_lr:.6f}|耗时:{time.time() - start_time:.2f}s")
            start_time=time.time()

    print("训练完成")

    # 1. 定义模型保存的目录路径 (这是一个字符串)
    model_save_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    
    # 2. 确保这个目录存在 (这个函数返回 None)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 3. 清理实验名称，使其适合作为文件名
    #    注意：这里的 .replace(" ", "_") 是为了处理原始字典中可能带空格的键
    sanitized_name = experiment_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    model_filename = f"addition_transformer_{sanitized_name}.pth"
    
    # 4. 使用 os.path.join 组合路径 (现在 model_save_dir 是一个字符串，所以这会正常工作)
    final_model_path = os.path.join(model_save_dir, model_filename)

    # 5. 保存模型
    torch.save(model.state_dict(), final_model_path)
    print(f"模型已经保存至 {final_model_path}")
    # 绘制损失曲线
    plot_losses(train_losses, val_losses, eval_iters_list,
                title=f'Loss Curve for Experiment: {experiment_name}',
                save_path=f"plots/addition_loss_{experiment_name}.png")

    return model

def run_evaluation(model, tokenizer, test_digit_pairs, num_test_cases=200, max_digits=4):
    print("\n"+"="*20+"开始评估"+"="*20)
    model.eval()
    eos_id=tokenizer.stoi['\n']
    pad_id=config.PAD_TOKEN_ID
    accuracies={}
    for d1, d2 in test_digit_pairs:
        correct_count=0
        print(f"\n---测试{d1}位数+{d2}位数的加法---")

        for i in range(num_test_cases):
            # 随机生成两个数字 n1 和 n2
            n1=random.randint(10**(d1-1) if d1>0 else 0, 10**d1-1)
            n2=random.randint(10**(d2-1) if d2>0 else 0, 10**d2-1)
            correct_answer_str=str(n1+n2)
            # 反转并填充数字部分，确保对齐
            s1_padded=str(n1).rjust(max_digits, config.PAD_TOKEN)
            s2_padded=str(n2).rjust(max_digits, config.PAD_TOKEN)

            # 构建输入提示
            prompt = f"{s1_padded}+{s2_padded}="
            context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.DEVICE).unsqueeze(0)
            # 生成模型输出
            generated_tokens = model.generate(context, config.MAX_NEW_TOKENS, eos_token_id=eos_id)[0].tolist()
            answer_ids=generated_tokens[(len(tokenizer.encode(prompt))):]

            cleaned_ids=[]
            for token_id in answer_ids:
                if token_id==eos_id or token_id==pad_id:
                    break
                cleaned_ids.append(token_id)
            model_answer_rev=tokenizer.decode(cleaned_ids)
            # 替换PAD_TOKEN并反转生成的答案
            model_answer_normal=model_answer_rev.replace(config.PAD_TOKEN, '')[::-1]

            # 比较反转后的答案
            if model_answer_normal == correct_answer_str:
                correct_count += 1

            original_prompt = f"{n1}+{n2}"
            status = "正确" if model_answer_normal == correct_answer_str else f"错误 (正确: {correct_answer_str})"
            print(f"  问题: {original_prompt:<12} 模型回答: {model_answer_normal:<8} -> {status}")

        accuracy = (correct_count / num_test_cases) * 100
        accuracies[f"{d1}位数 + {d2}位数"] = accuracy
        print(f"-> 结果: {d1}位数 + {d2}位数 准确率: {accuracy:.2f}% ({correct_count}/{num_test_cases})")
    print("=" * 50)
    model.train()  # 评估结束后切回训练模式
    return accuracies
def main():
    torch.manual_seed(42)
    random.seed(42)

    tokenizer=MathTokenizer(config.VOCAB)

    experiments = {
        # --- 实验 1: 综合基准 (Comprehensive Baseline) ---
        # 假设: 在一个覆盖1到4位数的丰富、多样化的数据集上训练，模型能够掌握所有分布内的加法组合。
        # 这是“最佳情况”的基准。
        "1_Comprehensive_Baseline": {
            "train_pairs": [
                (1, 1), (2, 2), (3, 3), (4, 4), # 对称
                (1, 3), (2, 4), # 不对称
            ],
            "test_pairs":  [
                (4, 4),      # 分布内，验证学习效果
                (3, 4),      # 分布内，但组合未见，测试组合泛化
            ]
        },

        # --- 实验 2: 长度插值 (Length Interpolation) ---
        # 假设: 如果模型真正学习了“算法”，它应该能解决从未见过的、介于训练数据长度“间隙”中的问题。
        "2_Length_Interpolation": {
            # 训练一个“有间隙”的分布：只包含1位数和4位数的问题
            "train_pairs": [
                (1, 2), (2, 2), (4, 4), (3, 4)
            ],
            # 测试模型能否填补(2,3)位数的“空隙”
            "test_pairs":  [(2, 2), (3, 3), (2, 3)]
        },

        # --- 实验 3: 从对称到非对称的结构泛化 ---
        # 假设: 一个只在对称问题 (d+d) 上训练的模型，可能无法泛化到结构不同的非对称问题 (d1+d2)。
        # 这旨在揭示模型学习的是表面模式还是深层算法。
        "3_Symmetry_To_Asymmetry_Generalization": {
            # 训练数据仅覆盖对称问题
            "train_pairs": [(1, 1), (2, 2), (3, 3), (4, 4)],
            # 测试模型能否处理从未见过的非对称结构
            "test_pairs":  [(2, 3), (1, 4)]
        }
    }
    all_results=collections.defaultdict(dict)

    for name, params in experiments.items():
        #生成数据
        print(f"\n{'='*20}正在为实验'{name}'生成数据{'='*20}")
        full_text_data=generate_dataset(20000, params['train_pairs'])
        full_dataset=AdditionDataset(full_text_data, tokenizer, config.BLOCK_SIZE)
        #划分训练和验证集
        train_size=int(0.9*len(full_dataset))
        val_size=len(full_dataset)-train_size
        train_dataset, val_dataset=random_split(full_dataset, [train_size, val_size])
        #创建DataLoader
        train_loader=DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader=DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )
        print("数据生成完成")

        #训练模型
        trained_model=run_training(train_loader, val_loader, name.replace(" ", "_"))

        #评估模型
        accuracies=run_evaluation(trained_model, tokenizer, params["test_pairs"])

        #保存结果
        for test_case, acc in accuracies.items():
            all_results[test_case][name]=acc

    print("\n=======所有实验完成，生成最终对比图========")
    plot_addition_accuracy_comparison(all_results)


if __name__ == "__main__":
    main()
