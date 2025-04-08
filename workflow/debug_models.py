import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import platform
import time


model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')


model.export(format="onnx", dynamic=True, simplify=True)