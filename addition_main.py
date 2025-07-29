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
from addition_data_utils import MathTokenizer, generate_dataset, get_batch
from addition_model import AdditionTransformer
from plot_utils import plot_losses
from plot_utils import plot_losses, plot_addition_accuracy_comparison
def run_training(train_data, val_data, experiment_name):
    print(f"\n========开始实验:{experiment_name}====")
    model=AdditionTransformer()
    model.to(config.DEVICE)
    print(f"模型参数量:{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer=optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DEACY)
    #定义warmup步数
    warmup_steps=int(config.TRAIN_STEPS*0.1)
    warmup_scheduler=LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler=CosineAnnealingLR(optimizer, T_max=config.TRAIN_STEPS-warmup_steps, eta_min=config.LEARNING_RATE/10)
    #余弦退火调度器
    scheduler=SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    train_losses, val_losses, eval_iters_list=[], [], []

    print("开始训练")
    start_time=time.time()
    for step in range(config.TRAIN_STEPS):

        xb, yb=get_batch('train', train_data, val_data)

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
                for _ in range(10):
                    x, y=get_batch('val', train_data, val_data)
                    _, val_loss=model(x, y)
                    val_loss_avg+=val_loss.item()
            model.train()
            val_loss_avg/=10
            train_losses.append(loss.item())
            val_losses.append(val_loss_avg)
            eval_iters_list.append(step)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"实验{experiment_name} 步数{step}/{config.TRAIN_STEPS}|训练损失:{loss.item():.4f}|验证损失:{val_loss_avg:.4f}|学习率:{current_lr:.6f}|耗时:{time.time() - start_time:.2f}s")
            start_time=time.time()

    print("训练完成")

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"模型已经保存至{config.MODEL_SAVE_PATH}")

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

    experiments={
        "Standard (Interpolation/Extrapolation)":{
            "train_pairs":[(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)],
            "test_pairs":[(1, 2), (1, 1), (2, 2)]
        },
        # "Hard Extrapolation":{
        #     "train_pairs":[(1, 1), (1, 2), (2, 2)],
        #     "test_pairs":[(3, 3), (4, 4)]
        # },
        # "Interpolation":{
        #     "train_pairs":[(2, 2), (4, 4)],
        #     "test_pairs":[(3, 3), (4, 4)]
        # }
    }
    all_results=collections.defaultdict(dict)

    for name, params in experiments.items():
        #生成数据
        print(f"\n{'='*20}正在为实验'{name}'生成数据{'='*20}")
        full_text_data=generate_dataset(20000, params['train_pairs'])
        data=torch.tensor(tokenizer.encode(full_text_data), dtype=torch.long)
        n=int(0.9*len(data))
        train_data, val_data=data[:n], data[n:]
        print("数据生成完成")

        #训练模型
        trained_model=run_training(train_data, val_data, name.replace(" ", "_"))

        #评估模型
        accuracies=run_evaluation(trained_model, tokenizer, params["test_pairs"])

        #保存结果
        for test_case, acc in accuracies.items():
            all_results[test_case][name]=acc

    print("\n=======所有实验完成，生成最终对比图========")
    plot_addition_accuracy_comparison(all_results)
    print("正在生成训练和验证数据")


if __name__ == "__main__":
    main()
