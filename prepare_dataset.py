import os
from datasets import load_dataset

def prepare_wikitext_dataset():
    """
    下载并处理 Wikitext-103-raw-v1 数据集。
    将所有分片（train, validation, test）合并到一个大的文本文件中，
    以便于我们的项目离线使用。
    """
    # 定义保存路径
    save_dir = "data"
    file_path = os.path.join(save_dir, "wikitext-103-raw.txt")
    
    # 如果文件已存在，则跳过
    if os.path.exists(file_path):
        print(f"数据集文件已存在于: {file_path}")
        return

    print("正在从 Hugging Face Hub 下载 wikitext-103-raw-v1 数据集...")
    # 加载数据集
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    print("数据集下载完成，正在处理并合并所有分片...")
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将所有分片的文本合并写入一个文件
    with open(file_path, "w", encoding="utf-8") as f:
        for split in ['train', 'validation', 'test']:
            print(f"正在处理 '{split}' 分片...")
            # 过滤掉空的或仅包含空格的行
            lines = [line for line in dataset[split]['text'] if line.strip()]
            f.write("\n".join(lines))
            # 在分片之间添加换行符以确保分隔
            f.write("\n")
            
    print(f"Wikitext 数据集已成功保存到: {file_path}")

if __name__ == "__main__":
    prepare_wikitext_dataset()