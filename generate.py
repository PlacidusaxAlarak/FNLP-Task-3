import torch
import config
from model import LanguageModel
from data_utils import load_data_and_prepare

def main():
    print("加载数据和分词器")
    _, _, tokenizer=load_data_and_prepare()

    print("加载模型")
    model=LanguageModel()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    print("开始生成文本")
    start_string="\n"
    context_tokens=tokenizer.encode(start_string)
    context=torch.tensor(context_tokens, dtype=torch.long, device=config.DEVICE).unsqueeze(0)#添加一个维度

    #生成
    with torch.no_grad():
        generated_tokens=model.generate(context, max_new_tokens=config.MAX_NEW_TOKENS)[0].tolist()#取出第一个

    #解码并且打印
    generated_text=tokenizer.decode(generated_tokens)
    print("---生成结果---")
    print(generated_text)
    print("----------------")

if __name__ == "__main__":
    main()