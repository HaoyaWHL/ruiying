# ruiying


## 1、现阶段进展

### 1。1、标签使用情况
当前多使用了Quartzity这个label，其他论文里没有在用

需要修改../data/下对应的images和labels（第一列的数字对应XXX标签）

### 1.2、样本数量
训练路径下，当前images=4000张，正好对应下面的Images=4000，但是每个图里可以有多个labels

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      75.8G      1.807      1.385      1.222        197        640: 100%|██████████| 32/32 [00:24<00:00,  1.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 16/16 [00:07<00:00,  2.19it/s]
                   all       4000       9211      0.401      0.525      0.479       0.21

考虑到数据总量非常庞大，需要分析confusion_matrix，并且可以进行样本选择

### 1.3、训练情况

对比论文里的一些参数设定，结合F1 score，当前learning_rate=1e-4偏低，decay=0.005偏大，训练应该是欠拟合，还有提升空间

