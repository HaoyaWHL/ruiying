import warnings
warnings.filterwarnings('ignore')
import platform
import time
import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')
import torch
import os

gpu_index = '3'
if gpu_index != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index


def demo_train():
    from ultralytics import YOLO

    model = YOLO('./tmp/yolov8.yaml')

    if platform.system() == 'Windows':
        data_config_path = './woodsurface.yaml'
    else:
        data_config_path = './linux_woodsurface.yaml'

    # return

    model.train(
        data=data_config_path,
        cache=True,
        imgsz=640,
        epochs=1,
        batch=10,
        close_mosaic=10,
        device=gpu_index,  # 使用 GPU 设备，如果有多个 GPU 可以设置为 '0,1'
        optimizer='SGD',  # 使用 SGD 优化器
        project='demo/train',
        name='exp',
    )

# demo_train()


# 主要是想知道怎么处理数据的，然后处理数据这块代码能否复用
def base():
    from ultralytics.models.yolo.detect import DetectionTrainer
    args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=1)
    trainer = DetectionTrainer(overrides=args)
    # dataloader = trainer.train()
    trainer.train()
    # trainer._setup_train()

    # for idx,data in enumerate(dataloader):
    #     print(data)
    #     break

# base()


def modify_backobone_from_deepseek_win():
    '''
    可以在windows环境下跑通，在deepseek问答给出的解决方案，直接用timm替换backbone的操作是可行的，能训练成功
    :return:
    '''
    import timm
    from ultralytics import YOLO

    # 自定义 backbone（例如 ResNet50）
    backbone = timm.create_model('resnet50', features_only=True, pretrained=True)

    # 修改 YOLO 的 backbone
    model = YOLO('yolov8n.yaml')  # 加载默认配置
    model.model.backbone = backbone  # 替换 backbone

    print(model)


    rng = time.strftime("%Y%m%d%H%M%S", time.localtime())
    print(rng)

    if platform.system() == 'Windows':
        data_config_path = './woodsurface.yaml'
    else:
        data_config_path = './linux_woodsurface.yaml'

    # return

    model.train(
        data=data_config_path,
        cache=True,
        imgsz=640,
        epochs=1,
        batch=10,
        close_mosaic=10,
        device='cpu',  # 使用 GPU 设备，如果有多个 GPU 可以设置为 '0,1'
        optimizer='SGD',  # 使用 SGD 优化器
        project='add_module/train',
        name='exp',
    )

def modify_backobone_from_deepseek_linux():
    '''
    直接从上面的代码迁移下来，准备在linux环境上跑，看能否有一个结果
    :return:
    '''
    def check_gpu_memory():
        if torch.cuda.is_available():
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    check_gpu_memory()

    import timm
    import sys
    sys.path.append("./")
    from ultralytics import YOLO

    # 自定义 backbone（例如 ResNet50）
    # 加载本地预训练模型
    # 假设模型文件路径为 './resnet50.pth'

    # backbone = timm.create_model('resnet50', features_only=True, pretrained=False)
    # backbone.load_state_dict(torch.load('./resnet50.pth'))

    def get_backbone_resnet():
        # load_resnet -- 可以跑，唯一的问题是需要过滤fc层（没办法直接下载，网络不通畅，就通过这种办法来做）
        # # 方法1：过滤不需要的键
        backbone = timm.create_model('resnet50', features_only=True, pretrained=False)
        state_dict = torch.load('./resnet50.pth')
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        backbone.load_state_dict(filtered_state_dict, strict=False)
        # 验证加载成功
        print("Backbone successfully loaded!")
        print("Number of parameters:", sum(p.numel() for p in backbone.parameters()))
        return backbone

    def get_backbone_mobilenetv4(name):
        # load mobilenetv4_conv_small
        #backbone = timm.create_model('mobilenetv4_conv_small', features_only=True, pretrained=False)
        #state_dict = torch.load('./mobilenetv4_conv_small.pth')
        # filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

        backbone = timm.create_model(name, features_only=True, pretrained=False)
        state_dict = torch.load(name + ".pth")

        backbone.load_state_dict(state_dict, strict=False)
        return backbone

    pre_name = "./pretrain_models/"
    # backbone_name = "mobilenetv4_conv_large"
    backbone_name = pre_name + "mobilenetv4_conv_medium"
    backbone = get_backbone_mobilenetv4(backbone_name)

    # 修改 YOLO 的 backbone
    #model = YOLO('yolov8n.yaml')  # 加载默认配置
    #model = YOLO('ultralytics/cfg/models/11/yolo11l.yaml') # 在workflow下加载模型的方式
    model = YOLO("./tmp/yolo11s.yaml")
    #model = YOLO('yolov11l.yaml')  # 加载默认配置
    model.model.backbone = backbone  # 替换 backbone

    print(model)

    model.train(
        # data='./linux_woodsurface.yaml',
        data='./linux_woodsurface_exclude_0.yaml',
        cache=True,  # 改为True可以加速数据加载(确保有足够RAM)
        imgsz=640,
        epochs=200,  # 木材缺陷可能需要更多epochs
        batch=128,  # 512对于大多数显卡过大，建议从32开始逐步增加
        close_mosaic=10,  # 提前关闭mosaic增强
        device=gpu_index,  # 考虑使用多GPU如'0,1,2,3'
        optimizer='SGD',  # 对于小数据集AdamW可能更好
        lr0=1e-3,  # 明确设置学习率
        weight_decay=1e-4,  # 添加权重衰减
        project='runs/train',
        name='large_model',  # 使用更有意义的名称
        #patience=20,  # 早停机制
        mixup=0.2,  # 数据增强
        hsv_h=0.015,  # 色相增强
        hsv_s=0.5,  # 饱和度增强
        hsv_v=0.3,  # 明度增强
        flipud=0.3,  # 上下翻转增强
        fliplr=0.4,  # 左右翻转增强
        #degrees=20.0,  # 旋转增强
        translate=0.1,  # 平移增强
        #scale=0.5,  # 缩放增强
        #shear=2.0,  # 剪切增强
        #perspective=0.00105,  # 透视变换
        save=True,  # 保存训练结果
        save_period=10,  # 每5个epoch保存一次
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
    if 0:
        # 使用DySample之后，onnx导出时会报错，因此暂时屏蔽
        model.export(
            format="onnx",
            dynamic=True,  # 动态输入
            simplify=True,  # 简化模型
            opset=12,  # ONNX opset版本
            imgsz=[640, 640],  # 输入尺寸
            batch=1  # 批处理大小
        )

modify_backobone_from_deepseek_linux()

def check_timm_models():
    import timm
    from ultralytics import YOLO
    all_models = timm.list_models()
    print(all_models)

    for i in all_models:
        if "mobilenetv4" in i:
            print(i)
    # mobilenetv4_conv_aa_large
    # mobilenetv4_conv_aa_medium
    # mobilenetv4_conv_blur_medium
    # mobilenetv4_conv_large
    # mobilenetv4_conv_medium
    # mobilenetv4_conv_small
    # mobilenetv4_conv_small_035
    # mobilenetv4_conv_small_050
    # mobilenetv4_hybrid_large
    # mobilenetv4_hybrid_large_075
    # mobilenetv4_hybrid_medium
    # mobilenetv4_hybrid_medium_075
# check_timm_models()


def download_resnet50_locally():
    import timm
    import torch

    # # 创建ResNet50模型并下载预训练权重
    # # model = timm.create_model('mobilenetv4_conv_small', pretrained=True)
    # model = timm.create_model('mobilenetv4_conv_large', pretrained=True)
    #
    # # 保存模型权重到本地
    # torch.save(model.state_dict(), './mobilenetv4_conv_large.pth')
    # # torch.save(model.state_dict(), './mobilenetv4_conv_small.pth')
    # # print("ResNet50模型权重已保存到本地: ./mobilenetv4_conv_small.pth")
    # print("模型权重已保存到本地")

    # modelist = ["mobilenetv4_conv_aa_large", "mobilenetv4_conv_aa_medium", "mobilenetv4_conv_blur_medium",
    #             "mobilenetv4_conv_large", "mobilenetv4_conv_medium", "mobilenetv4_conv_small",
    #             "mobilenetv4_conv_small_035", "mobilenetv4_conv_small_050", "mobilenetv4_hybrid_large",
    #             "mobilenetv4_hybrid_large_075", "mobilenetv4_hybrid_medium", "mobilenetv4_hybrid_medium_075"]
    modelist = ["mobilenetv4_conv_aa_large", "mobilenetv4_conv_aa_medium", "mobilenetv4_conv_blur_medium",
                "mobilenetv4_conv_medium",
                "mobilenetv4_conv_small_035", "mobilenetv4_conv_small_050", "mobilenetv4_hybrid_large",
                "mobilenetv4_hybrid_large_075", "mobilenetv4_hybrid_medium", "mobilenetv4_hybrid_medium_075"]
    for name in modelist:
        try:
            # 创建ResNet50模型并下载预训练权重
            model = timm.create_model(name, pretrained=True)

            # 保存模型权重到本地
            torch.save(model.state_dict(), './pretrain_models/{}.pth'.format(name))
            print("{}模型权重已保存到本地".format(name))
        except:
            print(name)
            continue

# download_resnet50_locally()





