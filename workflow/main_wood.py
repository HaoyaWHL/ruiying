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


def demo_train():
    rng = time.strftime("%Y%m%d%H%M%S", time.localtime())
    print(rng)
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

    if platform.system() == 'Windows':
        data_config_path = './woodsurface.yaml'
    else:
        data_config_path = './linux_woodsurface.yaml'

    model.train(
        data=data_config_path,
        cache=True,
        imgsz=640,
        epochs=1,
        batch=1,
        close_mosaic=10,
        device='cpu',  # 使用 GPU 设备，如果有多个 GPU 可以设置为 '0,1'
        optimizer='SGD',  # 使用 SGD 优化器
        project='runs/train',
        name='exp',
    )

    # Export the model
    # model.export(format="onnx")

def predict():
    from ultralytics import YOLO

    # 加载训练好的模型
    model = YOLO('./yolo11.onnx')

    import cv2
    import numpy as np

    # 加载要预测的图片
    image_path = r'E:\pycharm\github_reps\online_git\ruiying\data\woodsurface\images\train\99100003.jpg'
    image = cv2.imread(image_path)

    # 处理起来有点麻烦
    # image_label_path = image_path.replace("images","labels").replace(".jpg",".txt")


    # 将图片转换为 RGB 格式（YOLO 模型需要 RGB 格式）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行预测
    results = model.predict(image_rgb)

    # print(results)
    # print(model)

    # 进行预测
    # results = model(image_rgb)

    # 处理预测结果
    for det in results[0]:
        # 提取边界框坐标和类别索引
        x1, y1, x2, y2, confidence, cls = det.cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 获取类别名称
        class_name = model.names[int(cls)]

        # 绘制边界框和类别名称
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)

    # 显示结果
    cv2.imshow('Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


    # predict()

