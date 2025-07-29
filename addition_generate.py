import torch
import random
import os

try:
    import addition_config as config
    from addition_model import AdditionTransformer
    from addition_data_utils import MathTokenizer
except ImportError:
    print("错误：请确保 addition_config.py, addition_model.py, 和 addition_data_utils.py 文件")
    print("与此脚本位于同一目录下。")
    exit()


def load_model(model_path, device):
    """加载训练好的模型状态字典"""
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'。")
        print("这可能是因为：")
        print("1. 你还没有运行训练脚本 (addition_main.py)。")
        print("2. 模型保存的路径与 config 文件中的 MODEL_SAVE_PATH 不符。")
        exit()

    model = AdditionTransformer()
    model.to(device)

    # --- 核心修改：采纳 FutureWarning 的建议，并处理尺寸不匹配问题 ---
    try:
        # 使用 weights_only=True 更安全
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("\n" + "=" * 60)
        print("!!! 模型加载失败：尺寸不匹配 !!!")
        print("=" * 60)
        print("错误信息:", e)
        print("\n这通常意味着你正在尝试加载一个用旧配置训练的模型。")
        print("解决方案:")
        print("1. (推荐) 使用最新的 `addition_main.py` 重新训练一个模型。")
        print("2. 或者，将 `addition_config.py` 文件恢复到训练该模型时的状态。")
        print("=" * 60)
        exit()
    # --- 修改结束 ---

    print(f"模型已从 '{model_path}' 成功加载。")
    model.eval()
    return model


def generate_single_prediction(model, tokenizer, n1, n2, max_digits):
    """对单个加法问题进行预测"""
    correct_answer_str = str(n1 + n2)
    correct_answer_rev = correct_answer_str[::-1]

    s1_rev_padded = str(n1)[::-1].ljust(max_digits, config.PAD_TOKEN)
    s2_rev_padded = str(n2)[::-1].ljust(max_digits, config.PAD_TOKEN)
    prompt = f"{s1_rev_padded}+{s2_rev_padded}="

    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.DEVICE).unsqueeze(0)
    eos_id = tokenizer.stoi['\n']

    generated_tokens = model.generate(context, max_new_tokens=config.MAX_NEW_TOKENS, eos_token_id=eos_id)[0].tolist()

    model_output_padded = tokenizer.decode(generated_tokens)[len(prompt):].strip('\n')
    model_answer_rev = model_output_padded.replace(config.PAD_TOKEN, '')
    model_answer_str_for_display = model_answer_rev[::-1]

    is_correct = (model_answer_rev == correct_answer_rev)

    return {
        "prompt": f"{n1}+{n2}=",
        "model_answer": model_answer_str_for_display,
        "correct_answer": correct_answer_str,
        "is_correct": is_correct
    }


def main():
    test_pairs = [
        (2, 77),
        (9, 85),
        (1, 84),
        (5, 55),
        (45, 11),
        (85, 11)
    ]

    # 确保这个值与训练时一致
    max_digits_for_padding = 4

    model_path = config.MODEL_SAVE_PATH

    print("=" * 50)
    print(" 加法模型推理脚本")
    print("=" * 50)

    tokenizer = MathTokenizer(config.VOCAB)
    model = load_model(model_path, config.DEVICE)

    correct_count = 0
    total_count = len(test_pairs)

    print(f"\n开始对 {total_count} 个问题进行测试...\n")

    for n1, n2 in test_pairs:
        result = generate_single_prediction(model, tokenizer, n1, n2, max_digits_for_padding)

        status = "✅ 正确" if result["is_correct"] else "❌ 错误"
        if result["is_correct"]:
            correct_count += 1
            print(f"问题: {result['prompt']:<15} 模型回答: {result['model_answer']:<10} -> {status}")
        else:
            print(
                f"问题: {result['prompt']:<15} 模型回答: {result['model_answer']:<10} -> {status} (正确答案: {result['correct_answer']})")

    print("\n" + "=" * 50)
    print(" 推理完成")
    print(f" 总准确率: {correct_count / total_count:.2%} ({correct_count}/{total_count})")
    print("=" * 50)


if __name__ == "__main__":
    main()