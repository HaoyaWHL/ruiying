# encoding: utf-8

import torchvision
import torchvision.transforms as transforms

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any


class COCO8Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        COCO8 数据集加载器

        参数:
            root_dir: 数据集根目录 (包含 images/ 和 labels/ 文件夹)
            split: 'train' 或 'val'
            transform: 可选的图像变换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # 检查 split 是否有效
        assert split in ['train', 'val'], f"split 必须是 'train' 或 'val', 但得到 {split}"

        # 设置图像和标注路径
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)

        # 获取所有图像文件名 (不带扩展名)
        self.image_files = [f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

        # 验证图像和标注文件是否匹配
        self._validate_files()

    def _validate_files(self):
        """验证图像和标注文件是否一一对应"""
        for img_id in self.image_files:
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            label_path = os.path.join(self.label_dir, f"{img_id}.txt")

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"标注文件不存在: {label_path}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        获取一个样本

        返回:
            image: 图像张量 (C, H, W)
            target: 包含标注的字典，包括:
                - boxes: 边界框 [xmin, ymin, xmax, ymax]
                - labels: 类别标签
                - image_id: 图像ID
        """
        # 获取图像ID
        img_id = self.image_files[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

        # 加载标注
        label_path = os.path.join(self.label_dir, f"{img_id}.txt")
        boxes, labels = self._parse_label_file(label_path, image.shape)

        # 准备目标字典
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.as_tensor([idx], dtype=torch.int64)
        }

        # 应用变换 (如果有)
        if self.transform is not None:
            image, target = self.transform(image, target)

        # 转换为张量并返回
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
        return image, target

    def _parse_label_file(self, label_path: str, img_shape: Tuple[int, int, int]) -> Tuple[
        List[List[float]], List[int]]:
        """
        解析YOLO格式的标注文件

        参数:
            label_path: 标注文件路径
            img_shape: 图像形状 (H, W, C)

        返回:
            boxes: 边界框列表 [xmin, ymin, xmax, ymax]
            labels: 类别标签列表
        """
        img_h, img_w = img_shape[:2]
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # 跳过无效行

                # 解析类别和边界框 (YOLO格式: class x_center y_center width height)
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h

                # 转换为 [xmin, ymin, xmax, ymax]
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

        return boxes, labels


def collate_fn(batch):
    """
    自定义批处理函数，处理可变数量的目标

    参数:
        batch: 一个批次的 (image, target) 元组

    返回:
        images: 图像张量 (B, C, H, W)
        targets: 目标字典列表
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


def get_dataloaders(root_dir: str, batch_size: int = 4, num_workers: int = 0, transform=None) -> Tuple[
    DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器

    参数:
        root_dir: 数据集根目录
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        transform: 可选的图像变换

    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建数据集
    train_dataset = COCO8Dataset(root_dir, split='train', transform=transform)
    val_dataset = COCO8Dataset(root_dir, split='val', transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


# 示例用法
if __name__ == "__main__":
    # 设置数据集路径
    root_dir = "./datasets/coco8"  # 替换为你的实际路径

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(root_dir, batch_size=2)

    # 打印数据集信息
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")

    # 获取一个批次并检查形状
    images, targets = next(iter(train_loader))
    print(f"图像批次形状: {images.shape}")  # 应该是 (batch_size, 3, H, W)
    print(f"第一个目标的边界框: {targets[0]['boxes']}")
    print(f"第一个目标的标签: {targets[0]['labels']}")