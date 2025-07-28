import torch

from config import MAX_NEW_TOKENS

VOCAB="0123456789+=\n"
VOCAB_SIZE=len(VOCAB)

#模型架构配置
BLOCK_SIZE=32
D_MODEL=256
N_HEADS=8
N_LAYERS=8
DROPOUT=0.5
WEIGHT_DEACY=0.1
#训练配置
BATCH_SIZE=256
LEARNING_RATE=1e-3
TRAIN_STEPS=8000
EVAL_INTERVAL=200#每N步验证一次
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS=10
#推理评估配置
MODEL_SAVE_PATH="models/addition_transformer.pth"
PLOT_SAVE_PATH="plots/addition_loss_curve.png"

print(f"设备: {DEVICE}")
print(f"词汇表大小:{VOCAB_SIZE}")