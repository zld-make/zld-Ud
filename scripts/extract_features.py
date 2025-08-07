import os
import sys

# 获取当前脚本的绝对路径（假设脚本位于 scripts/ 目录）
current_script_path = os.path.abspath(__file__)
# 定位项目根目录（假设项目结构为 U-ViT-main/scripts/extract_features.py）
project_root = os.path.dirname(os.path.dirname(current_script_path))  # 上两级目录
sys.path.insert(0, project_root)  # 将项目根目录添加到 Python 路径
# print("当前 Python 路径:", sys.path)
# print("项目根目录:", project_root)
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm



def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='输入数据集根目录路径')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = f'D:/Desktop/DM/yuan/U-ViT-main/assets/datasets/custom{resolution}_features'
    os.makedirs(output_dir, exist_ok=True)

    # 数据集加载与预处理
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root=args.path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        persistent_workers=True
    )

    # 加载自动编码器
    model = get_model('D:\Desktop\DM\yuan/U-ViT-main/assets\stable-diffusion/autoencoder_kl.pth')
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    idx = 0
    for imgs, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            moments = model(imgs, fn='encode_moments').cpu().numpy()

        for moment, label in zip(moments, labels.numpy()):
            data = np.array((moment, label), dtype=object)
            np.save(os.path.join(output_dir, f'{idx}.npy'), data)
            idx += 1

    print(f'共处理 {idx} 个样本，特征保存在 {output_dir}')


if __name__ == "__main__":
    main()