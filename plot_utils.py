import matplotlib.pyplot as plt
import os
import config

def plot_losses(train_losses, val_losses, eval_iters):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_iters, val_losses, label='Validation Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    #确保保存路径的目录存在
    save_dir=os.path.dirname(config.PLOT_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(config.PLOT_SAVE_PATH)
    print(f"损失曲线图已保存至: {config.PLOT_SAVE_PATH}")
    plt.show()