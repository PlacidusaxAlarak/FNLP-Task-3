import torch
import config
from model import LanguageModel
from data_utils import load_data_and_prepare

def main():
    experiments=[
        {
            'name':'Char Tokenizer',
            'model_path':'models/lm_Char_Tokenizer.pth',
            'tokenizer_type':'char',
            'vocab_size':None
        },
        {
            'name':'BPE(vocab=5000)',
            'model_path':'models/lm_BPE_vocab=5000.pth',
            'tokenizer_type':'bpe',
            'vocab_size':5000
        }
    ]
    for exp_conf in experiments:
        print("\n" + "#" * 20 + f"开始为实验{exp_conf['name']}生成文本" + "#" * 20)

        config.TOKENIZER_TYPE=exp_conf['tokenizer_type']
        config.VOCAB_SIZE=exp_conf['vocab_size']

        print(f"--- 使用配置: Tokenizer={config.TOKENIZER_TYPE}, Vocab Size={config.VOCAB_SIZE or '动态确定'} ---")

        #加载分词器
        _, _, tokenizer=load_data_and_prepare()

        #加载对应模型
        print("加载模型")
        model=LanguageModel()
        model_path=exp_conf['model_path']
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))

        model.to(config.DEVICE)
        model.eval()

        #生成文本
        print("开始生成")
        start_string="\nIt was a dark night"
        context_tokens=tokenizer.encode(start_string)
        context=torch.tensor(context_tokens, dtype=torch.long, device=config.DEVICE).unsqueeze(0)

        with torch.no_grad():
            generated_tokens=model.generate(context, max_new_tokens=config.MAX_NEW_TOKENS, temperature=0.7, top_k=None, top_p=0.9)[0].tolist()

        generated_text=tokenizer.decode(generated_tokens)
        print("\n" + "--- " + f"'{exp_conf['name']}' 的生成结果" + " ---")
        print(generated_text)
        print("-------------------------------------------------------\n")

if __name__ == "__main__":
    main()