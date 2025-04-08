# encoding:utf-8

import datasets
import os

# 指定要读取的目录路径
directory = r'../data/wood/Bounding Boxes - YOLO Format - 1/Bounding Boxes - YOLO Format - 1'

labels = set()
# # 遍历目录下的所有文件
for filename in os.listdir(directory):
#     # 检查是否为txt文件
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            # 在这里处理文件内容
            print(f"正在处理文件: {file_path}")
            for line in f.readlines():
                line = line.strip().split(" ")
                # print(line)
                labels.add(line[0])
print(labels)



# data_path = r'E:\pycharm\github_reps\online_git\ruiying\data\wood\Bounding Boxes - YOLO Format - 1\Bounding Boxes - YOLO Format - 1\99100003.txt'
# labels = set()
# with open(data_path, 'r', encoding='utf-8') as f:
#     # 在这里处理文件内容
#     print(f"正在处理文件: {data_path}")
#     for line in f.readlines():
#         line = line.strip().split(" ")
#         # print(line)
#         labels.add(line[0])
#     # break
# print(labels)

import cv2
import numpy as np
from ultralytics import YOLO

def detect_wood_defects(image_path, model_path):
    # 加载预训练的YOLO模型
    model = YOLO(model_path)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")

    # 执行目标检测
    results = model(image)

    # 在图像上绘制检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 获取置信度和类别
            conf = float(box.conf)
            cls = int(box.cls)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加标签文本
            label = f'缺陷{cls}: {conf:.2f}'
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('木材缺陷检测结果', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    image_path = "测试图片路径.jpg"
    model_path = "yolo模型路径.pt"
    detect_wood_defects(image_path, model_path)