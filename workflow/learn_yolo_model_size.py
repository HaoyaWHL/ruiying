import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import platform
import time
import torch

# model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
# Export the model
# model.export(format="onnx")


# 初始化一个轻量级的 YOLOv8n 模型
# 从网络下载预训练的 YOLOv8n 模型
# model = YOLO('yolov8n.pt')  # 加载官方预训练的 nano 版本模型


from ultralytics import YOLO

# 向YOLOv5一样，根据参数数量，YOLOv8有5种不同类型的模型：nano(n), small(s), medium(m), large(l), and extra large(x)，如下图所示：

# Load a model
# 0.04 G
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# 0.04 G
# model = YOLO("yolo11s.yaml")  # build a new model from YAML
# 0.08G
# model = YOLO("yolo11m.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# 0.1G 0.22G
item = ['l', 'x']  # 当前使用large版本足够

for i in item:
    model = YOLO("yolo11" + i + ".yaml")  # build a new model from YAML

    # 将模型移动到GPU（如果有）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取模型的内存占用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.to(device)
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        print(f"Memory allocated: {memory_allocated:.2f} GB")
    else:
        print("No GPU available. Memory allocation check is not applicable.")

# model.export("./ckpt/yolo11n.onnx")
# model.export(format="onnx", dynamic=True, simplify=True)
