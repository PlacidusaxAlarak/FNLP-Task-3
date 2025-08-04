import torch
import re
import os
import addition_config as config
from addition_model import AdditionTransformer
from addition_data_utils import MathTokenizer

def format_problem_for_inference(n1:int, n2:int, max_digits:int=config.MAX_DIGITS)->str:
    #将两个数字格式化为模型推理所需的输入字符串
    s1_padded=str(n1).rjust(max_digits, config.PAD_TOKEN)
    s2_padded=str(n2).rjust(max_digits, config.PAD_TOKEN)
    prompt=f"{s1_padded}+{s2_padded}="
    return prompt

def postprocess_model_output(generated_ids:list[int], tokenizer:MathTokenizer)->str:
    eos_id=tokenizer.stoi['\n']
    pad_id=config.PAD_TOKEN_ID

    cleaned_ids=[]
    for token_id in generated_ids:
        if token_id==eos_id or token_id==pad_id:
            break
        cleaned_ids.append(token_id)

    #解码得到反向的答案字符串
    model_answer_reversed=tokenizer.decode(cleaned_ids)

    #再次反转以获得正确的数字顺序
    final_answer=model_answer_reversed[::-1]
    return final_answer
def main():
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"错误:在{config.MODEL_SAVE_PATH}找不到模型文件。请检查路径是否正确。")
        print("请运行addition_main.py训练模型并保存模型文件。")
        return

    print("正在加载模型和分词器")
    tokenizer=MathTokenizer(config.VOCAB)
    model=AdditionTransformer()

    #加载模型权重
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print("模型加载成功")

    print("\n=====加法运算模型推理====")
    print("请输入一个加法问题, 或输入quit退出")

    while True:
        user_input=input("问题:")
        if user_input.lower() in ['quit', 'exit']:
            print("再见！")
            break

        #解析和验证用户输入
        match = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', user_input)
        if not match:
            print("无效的输入格式。请输入两个数字，用加号连接。")
            continue

        n1_str, n2_str=match.groups()
        n1, n2=int(n1_str), int(n2_str)

        prompt=format_problem_for_inference(n1, n2, max_digits=config.MAX_DIGITS)
        context=torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.DEVICE).unsqueeze(0)

        #使用模型进行推理
        with torch.no_grad():
            generated_output=model.generate(
                idx=context,
                max_new_tokens=config.MAX_NEW_TOKENS,
                eos_token_id=tokenizer.stoi['\n']
            )

        #后处理模型输出
        all_generated_ids=generated_output[0].tolist()
        #仅提取新生成部分
        answer_ids_only=all_generated_ids[len(prompt):]

        model_answer=postprocess_model_output(answer_ids_only, tokenizer)
        correct_answer=str(n1+n2)

        print(f"  模型的回答: {model_answer}")
        print(f"  正确的答案: {correct_answer}")
        if model_answer == correct_answer:
            print("  结果: \033[92m正确\033[0m\n")  # 绿色
        else:
            print("  结果: \033[91m错误\033[0m\n")  # 红色

if __name__ == "__main__":
    main()