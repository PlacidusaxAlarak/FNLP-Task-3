import torch
import torch.optim as optim
import random
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import addition_config as config
from addition_data_utils import MathTokenizer, generate_dataset, get_batch
from addition_model import AdditionTransformer
from plot_utils import plot_losses

def run_training(train_data, val_data):
    model=AdditionTransformer()
    model.to(config.DEVICE)
    print(f"模型参数量:{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer=optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DEACY)

    train_losses, val_losses, eval_iters_list=[], [], []

    print("开始训练")
    start_time=time.time()
    for step in range(config.TRAIN_STEPS):
        xb, yb=get_batch('train', train_data, val_data)

        logits, loss=model(xb, yb)

        optimizer.zero_grad(set_to_none=True)#优化，不会重新分配内存
        loss.backward()#计算梯度
        optimizer.step()#更新权重

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
            print(f"步数{step}/{config.TRAIN_STEPS}|训练损失:{loss.item():.4f}|验证损失:{val_loss_avg:.4f}|耗时:{time.time()-start_time:.2f}s")
            start_time=time.time()

    print("训练完成")

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"模型已经保存至{config.MODEL_SAVE_PATH}")

    # 绘制损失曲线
    plot_losses(train_losses, val_losses, eval_iters_list)

    return model

def run_evaluation(model, tokenizer, test_digit_pairs, num_test_cases=200):
    print("\n"+"="*20+"开始评估"+"="*20)
    model.eval()
    eos_id=tokenizer.stoi['\n']

    for d1, d2 in test_digit_pairs:
        correct_count=0
        print(f"\n---测试{d1}位数+{d2}位数的加法---")

        for i in range(num_test_cases):
            n1=random.randint(10**(d1-1) if d1>0 else 0, 10**d1-1)
            n2=random.randint(10**(d2-1) if d2>0 else 0, 10**d2-1)
            correct_answer=str(n1+n2)

            prompt=f"{n1}+{n2}="
            context=torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.DEVICE).unsqueeze(0)

            generated_tokens=model.generate(context, config.MAX_NEW_TOKENS, eos_token_id=eos_id)[0].tolist()
            model_answer=tokenizer.decode(generated_tokens)[len(prompt):]

            if model_answer==correct_answer:
                correct_count+=1

            if i % (num_test_cases // 5) == 0:
                status = "正确" if model_answer == correct_answer else f"错误 (正确: {correct_answer})"
                print(f"  问题: {prompt:<12} 模型回答: {model_answer:<8} -> {status}")

        accuracy = (correct_count / num_test_cases) * 100
        print(f"-> 结果: {d1}位数 + {d2}位数 准确率: {accuracy:.2f}% ({correct_count}/{num_test_cases})")
        print("=" * 50)
        model.train()  # 评估结束后切回训练模式


def main():
    torch.manual_seed(42)
    random.seed(42)

    tokenizer=MathTokenizer(config.VOCAB)

    TRAIN_PAIRS=[(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]
    TEST_PAIRS=[(1, 1), (1, 2), (3, 3), (3, 4), (4, 4)]


    print("正在生成训练和验证数据")
    full_text_data=generate_dataset(20000, TRAIN_PAIRS)
    data=torch.tensor(tokenizer.encode(full_text_data), dtype=torch.long)
    n=int(0.9*len(data))
    train_data=data[:n]
    val_data=data[n:]
    print("数据生成完毕")

    train_model=run_training(train_data, val_data)

    run_evaluation(train_model, tokenizer, TEST_PAIRS)

if __name__ == "__main__":
    main()
