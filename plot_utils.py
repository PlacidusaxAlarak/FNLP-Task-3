# plot_utils.py
import matplotlib.pyplot as plt
import os

# 根据调用脚本确定config
try:
    import addition_config as config
except ImportError:
    import config as config
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

def plot_losses(train_losses, val_losses, eval_iters, title='Training vs Validation Loss', save_path=None):
    # 使函数更通用，可以接受标题和保存路径
    if save_path is None:
        save_path = config.PLOT_SAVE_PATH

    plt.figure(figsize=(10, 5))

    # 平滑训练损失使其更易于观察
    if len(train_losses) > 10:
        train_losses_smooth = np.convolve(train_losses, np.ones(10) / 10, mode='valid')
        plt.plot(train_losses_smooth, label='Smoothed Training Loss')
    else:
        plt.plot(train_losses, label='Training Loss')

    plt.plot(eval_iters, val_losses, label='Validation Loss', marker='o')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_path)
    print(f"损失曲线图已保存至: {save_path}")
    plt.close()  # 关闭图像，防止在Jupyter等环境中重复显示


def plot_addition_accuracy_comparison(results):
    """
    绘制加法任务在不同实验下的准确率对比条形图。
    results 格式: {'3+3': {'Exp1': 99.5, 'Exp2': 80.0}, '4+4': {'Exp1': 90.0, 'Exp2': 20.0}}
    """
    test_cases = list(results.keys())
    experiment_names = list(next(iter(results.values())).keys())

    n_groups = len(test_cases)
    n_bars = len(experiment_names)

    fig, ax = plt.subplots(figsize=(10, 7))

    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    for i, exp_name in enumerate(experiment_names):
        accuracies = [results[case].get(exp_name, 0) for case in test_cases]
        rects = ax.bar(index + i * bar_width, accuracies, bar_width,
                       alpha=opacity, label=exp_name)
        # 在条形图上显示数值
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xlabel('Test Case (d1+d2 digits)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Generalization on Addition Task')
    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(test_cases)
    ax.set_ylim(0, 110)
    ax.legend(title="Experiment Type")

    fig.tight_layout()

    # 保存图像
    save_path = "plots/addition_accuracy_comparison.png"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path)
    print(f"准确率对比图已保存至: {save_path}")
    plt.show()


def plot_lm_loss_comparison(results):
    """
    绘制语言模型在不同实验下的验证损失对比图。
    results 格式: {'Char': ([iters], [losses]), 'BPE_1000': ([iters], [losses])}
    """
    plt.figure(figsize=(12, 7))

    for name, (iters, losses) in results.items():
        plt.plot(iters, losses, label=f'Validation Loss ({name})', marker='o', markersize=3, linestyle='--')

    plt.title('Language Model Validation Loss Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 损失通常在对数尺度上更清晰

    save_path = "plots/lm_loss_comparison.png"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_path)
    print(f"语言模型损失对比图已保存至: {save_path}")
    plt.show()