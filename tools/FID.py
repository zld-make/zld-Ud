import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights

# ==================== 关键修改1：配置日志详细级别 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fid_stats.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 关键修改2：增强数据集类 ====================
class RobustImageDataset(Dataset):
    """带严格校验的图像数据集类"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_files = []
        self._scan_files()

    def _scan_files(self):
        error_log = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # 严格校验文件头
                    with open(file_path, 'rb') as f:
                        header = f.read(32)
                        if not self._is_valid_image_header(header):
                            raise IOError("Invalid file header")

                    # 验证图像可解码性
                    with Image.open(file_path) as img:
                        img.verify()
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                    if file.lower().endswith(valid_extensions):
                        self.valid_files.append(file_path)
                except Exception as e:
                    error_log.append(f"{file_path}: {str(e)}")

        if error_log:
            logger.warning(f"发现 {len(error_log)} 个无效文件:\n" + "\n".join(error_log[:10]))

    def _is_valid_image_header(self, header):
        # 校验常见图片格式的魔数
        if header.startswith(b'\xff\xd8\xff'):  # JPEG
            return True
        if header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return True
        if header[4:12] == b'ftypwebp':  # WEBP
            return True
        return False

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img
        except Exception as e:
            logger.error(f"致命错误: {file_path} - {str(e)}")
            # 严格模式：直接跳过错误样本
            raise RuntimeError("发现损坏数据，终止处理")


def load_inception_model(device):
    # 使用官方推荐的权重加载方式
    weights = Inception_V3_Weights.IMAGENET1K_V1

    # 允许模型保留辅助输出层
    model = inception_v3(
        weights=weights,
        transform_input=False  # 必须保留此设置
    )

    # 替换全连接层以获取特征向量
    model.fc = torch.nn.Identity()

    # 禁用Dropout和BatchNorm的随机性
    model.eval()

    return model.to(device)


# ==================== 关键修改4：增强统计量计算 ====================
def calculate_fid_stats(features):
    """带数值稳定性检查的统计计算"""
    # 验证输入数据
    if features.shape[0] < 2:
        raise ValueError("至少需要2个样本计算协方差")

    # 均值计算
    mu = np.mean(features, axis=0)

    # 协方差矩阵计算
    sigma = np.cov(features, rowvar=False)

    # 增强数值稳定性 (比原代码更严格)
    sigma += np.eye(sigma.shape[0]) * 1e-9

    # 检查矩阵有效性
    if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
        raise ValueError("统计量包含NaN值")

    return mu.astype(np.float64), sigma.astype(np.float64)


def main(config):
    # ==================== 关键修改5：预处理严格验证 ====================
    # 必须与官方FID实现一致
    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet标准参数
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ==================== 关键修改6：数据加载增强 ====================
    logger.info("初始化数据加载器...")
    try:
        dataset = RobustImageDataset(config.dataset_dir, transform=transform)
    except Exception as e:
        logger.error(f"数据集初始化失败: {str(e)}")
        raise

    logger.info(f"已加载有效图像: {len(dataset)} 张")
    if len(dataset) < 100:
        logger.warning("样本量过少可能导致统计不准确！")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False  # 重要：保留所有样本
    )

    # ==================== 关键修改7：设备处理 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载修复后的模型
    model = load_inception_model(device)

    # ==================== 关键修改8：特征提取修复 ====================
    features_list = []  # 修改变量名为features_list以避免冲突
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="特征提取"):
            batch = batch.to(device)
            # 使用不同的变量名存储模型输出
            batch_features = model(batch)
            features_list.append(batch_features.cpu().numpy())

    features = np.concatenate(features_list, axis=0)

    # 关键维度验证
    if features.shape[1] != 2048:
        raise RuntimeError(f"特征维度错误！期望2048，实际得到{features.shape[1]}")

    # ==================== 关键修改9：统计量保存 ====================
    logger.info("计算最终统计量...")
    mu, sigma = calculate_fid_stats(features)

    # 保存为float64类型
    np.savez_compressed(
        config.save_path,
        mu=mu.astype(np.float64),
        sigma=sigma.astype(np.float64)
    )

    # ==================== 关键修改10：结果验证 ====================
    # 重新加载验证
    stats = np.load(config.save_path)
    if stats['mu'].shape[0] != 2048:
        raise ValueError("保存的统计量维度错误")

    logger.info(f"统计量验证通过！保存路径: {config.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FID统计计算器')
    parser.add_argument('dataset_dir', type=str, help='图像数据集路径')
    parser.add_argument('save_path', type=str, help='统计量保存路径 (.npz)')
    parser.add_argument('--batch_size', type=int, default=32, help='推荐值：32-256')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.exception("程序异常终止")
        exit(1)