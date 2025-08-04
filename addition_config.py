import torch

from config import MAX_NEW_TOKENS
PAD_TOKEN="_"
VOCAB=f"0123456789+=\n{PAD_TOKEN}"
VOCAB_SIZE=len(VOCAB)
PAD_TOKEN_ID=tuple(VOCAB).index(PAD_TOKEN)

#模型架构配置
BLOCK_SIZE=32
D_MODEL=512
N_HEADS=16
N_LAYERS=16
DROPOUT=0.1
MAX_DIGITS=4
WEIGHT_DECAY=0.01
#训练配置
BATCH_SIZE=32
LEARNING_RATE=5e-4
TRAIN_STEPS=8000
EVAL_INTERVAL=200#每N步验证一次
DEVICE='cuda:1' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS=15
#推理评估配置
MODEL_SAVE_PATH="models/addition_transformer_1_Comprehensive_Baseline.pth"
PLOT_SAVE_PATH="plots/addition_loss_curve.png"
