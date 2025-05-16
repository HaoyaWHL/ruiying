# encoding: utf-8

# V2版本，下载更多数据，进行训练与测试
# https://datasetninja.com/wood-defect-detection#download

# 全是选了3600张，而且剔除了最少的3类标签数据，只和original algo做对比

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import platform
import time
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


def baseline():
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

    model.train(
        data='./linux_woodsurface.yaml',
        cache=False,
        imgsz=640,
        epochs=20,
        batch=512,
        close_mosaic=100,
        device='5',  # 使用 GPU 设备，如果有多个 GPU 可以设置为 '0,1'
        optimizer='SGD',  # 使用 SGD 优化器
        project='runs/train',
        name='exp',
    )

    # Export the model
    model.export(format="onnx")


def deepseek():
    '''由deepseek提供的代码'''
    def check_gpu_memory():
        if torch.cuda.is_available():
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    check_gpu_memory()

    # 加载模型 (推荐使用预训练权重)
    # model = YOLO('yolov8n.pt')  # 或者 yolov8s/m/l/x
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    model = YOLO("yolo11l.yaml")  # build a new model from YAML

    model.train(
        data='./linux_woodsurface.yaml',
        cache=True,  # 改为True可以加速数据加载(确保有足够RAM)
        imgsz=640,
        epochs=10,  # 木材缺陷可能需要更多epochs
        batch=128,  # 512对于大多数显卡过大，建议从32开始逐步增加
        close_mosaic=10,  # 提前关闭mosaic增强
        device='1,2',  # 考虑使用多GPU如'0,1,2,3'
        optimizer='AdamW',  # 对于小数据集AdamW可能更好
        lr0=1e-4,  # 明确设置学习率
        weight_decay=0.05,  # 添加权重衰减
        project='runs/train',
        name='large_model',  # 使用更有意义的名称
        patience=20,  # 早停机制
        mixup=0.2,  # 数据增强
        hsv_h=0.015,  # 色相增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        flipud=0.5,  # 上下翻转增强
        fliplr=0.5,  # 左右翻转增强
        degrees=10.0,  # 旋转增强
        translate=0.1,  # 平移增强
        scale=0.5,  # 缩放增强
        shear=2.0,  # 剪切增强
        perspective=0.0005,  # 透视变换
        save=True,  # 保存训练结果
        save_period=5,  # 每5个epoch保存一次
        single_cls=False,  # 如果是多类别检测
        # pretrained=True  # 使用预训练权重
    )

    # 验证
    metrics = model.val()
    print("\n\n\n")
    print(f"验证结果 - mAP50-95: {metrics.box.map:.4f}")
    print("\n\n\n")

    # 导出模型
    # 导出为ONNX时添加更多参数
    model.export(
        format="onnx",
        dynamic=True,  # 动态输入
        simplify=True,  # 简化模型
        opset=12,  # ONNX opset版本
        imgsz=[640, 640],  # 输入尺寸
        batch=1  # 批处理大小
    )


if __name__ == '__main__':
    deepseek()
