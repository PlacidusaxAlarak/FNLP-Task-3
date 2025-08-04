import torch

#数据和分词器配置
DATA_FILE_PATH="data/wikitext-103-raw.txt"
TOKENIZER_TYPE='char'
VOCAB_SIZE=None

#docoder-only模型架构
MODEL_TYPE='decoder-only'
BLOCK_SIZE=256
D_MODEL=1024
N_HEADS=64
N_LAYERS=16
DROPOUT=0.1

#训练配置sdf
BATCH_SIZE=32
LEARNING_RATE=3e-4
EPOCHS=5000
MAX_ITERS=40000
WEIGHT_DECAY=0.01
EVAL_INTERVAL=250#每N词迭代，进行一次验证
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS=16

#新增：学习率调度器配置
WARMUP_ITERS=200#预热的迭代次数
MIN_LR=3e-5#学习率退火后的最小值
LR_DECAY_ITERS=MAX_ITERS#学习率衰减的总迭代次数
#推理配置
MAX_NEW_TOKENS=500#生成文本的最大长度

#保存/加载路径
MODEL_SAVE_PATH="models/transformer_lm.pth"
PLOT_SAVE_PATH = f"plots/loss_curve_{TOKENIZER_TYPE}.png"

if __name__=='__main__':
    print(f"使用的设备是{DEVICE}")

