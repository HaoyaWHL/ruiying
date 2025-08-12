import timm
import torch
from PIL import Image
import torchvision.transforms as transforms

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import os
# from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import time

# 设置随机种子保证可重复性
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # 获取图像ID
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # 转换为PIL图像
        img = F.to_pil_image(img)

        # 转换标注格式
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for obj in target['annotations']:
            # COCO标注格式是[x_min, y_min, width, height]
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = xmin + obj['bbox'][2]
            ymax = ymin + obj['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])
            areas.append(obj['area'])
            iscrowd.append(obj['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

from torchvision import transforms as T

def get_transform(train):
    transforms = []
    # 转换为Tensor
    transforms.append(T.ToTensor())
    if train:
        # 训练时添加数据增强 (50%概率水平翻转)
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



# 加载数据集
def load_datasets(data_path):
    # 路径设置
    train_data_dir = os.path.join(data_path, "train2017")
    train_ann_file = os.path.join(data_path, "annotations", "instances_train2017.json")
    val_data_dir = os.path.join(data_path, "val2017")
    val_ann_file = os.path.join(data_path, "annotations", "instances_val2017.json")

    # 创建数据集
    train_dataset = CocoDetection(
        root=train_data_dir,
        annFile=train_ann_file,
        transforms=get_transform(train=True)
    )

    val_dataset = CocoDetection(
        root=val_data_dir,
        annFile=val_ann_file,
        transforms=get_transform(train=False)
    )

    return train_dataset, val_dataset


# 创建模型
def create_model(num_classes):
    # 加载预训练的backbone
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    # 定义anchor生成器
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # 定义ROI pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # 创建Faster R-CNN模型
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


# 训练函数
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


# 评估函数
def evaluate(model, data_loader, device):
    model.eval()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            model(images)


# 可视化函数
def visualize_predictions(model, dataset, device, num_images=3):
    model.eval()

    for i in range(num_images):
        # 随机选择一张图像
        idx = random.randint(0, len(dataset) - 1)
        image, _ = dataset[idx]

        with torch.no_grad():
            prediction = model([image.to(device)])

        image = image.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box, label, score in zip(
                prediction[0]['boxes'].cpu().numpy(),
                prediction[0]['labels'].cpu().numpy(),
                prediction[0]['scores'].cpu().numpy()
        ):
            if score > 0.5:  # 只显示置信度大于0.5的预测
                x, y, w, h = box
                rect = patches.Rectangle(
                    (x, y), w - x, h - y,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x, y,
                    f'{label}: {score:.2f}',
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )

        plt.show()


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 数据集路径 - 修改为你的COCO数据集路径
    data_path = "path/to/your/coco/dataset"

    # 加载数据集
    train_dataset, val_dataset = load_datasets(data_path)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # COCO有91类，但实际只有80类被使用
    num_classes = 91  # 包括背景

    # 创建模型
    model = create_model(num_classes)
    model.to(device)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 训练参数
    num_epochs = 5

    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        train_one_epoch(model, optimizer, train_loader, device, epoch)

        # 每个epoch后在验证集上评估
        evaluate(model, val_loader, device)

        print(f"Epoch completed in {time.time() - start_time:.2f} seconds")

    # 保存模型
    torch.save(model.state_dict(), "coco_detection_model.pth")
    print("Model saved to coco_detection_model.pth")

    # 可视化一些预测结果
    visualize_predictions(model, val_dataset, device)


if __name__ == "__main__":
    main()