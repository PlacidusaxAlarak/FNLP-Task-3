import torch

#数据和分词器配置
DATA_FILE_PATH="data/sherlock.txt"
TOKENIZER_TYPE='char'
VOCAB_SIZE=None

#docoder-only模型架构
MODEL_TYPE='decoder-only'
BLOCK_SIZE=128
D_MODEL=384
N_HEADS=6
N_LAYERS=6
DROPOUT=0.5

#训练配置
BATCH_SIZE=64
LEARNING_RATE=3e-4
EPOCHS=5000
EVAL_INTERVAL=250#每N词迭代，进行一次验证
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

#推理配置
MAX_NEW_TOKENS=500#生成文本的最大长度

#保存/加载路径
MODEL_SAVE_PATH="models/transformer_lm.pth"
PLOT_SAVE_PATH = f"plots/loss_curve_{TOKENIZER_TYPE}.png"

print(f"使用的设别是{DEVICE}")

