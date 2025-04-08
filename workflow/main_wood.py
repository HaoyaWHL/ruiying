import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import platform
import time

def train():
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

predict()

