# encoding: utf-8

import os

def read_all_txt_files(path):
    """读取指定路径下的所有txt文件"""
    txt_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files


def get_labels_st():
    '''
    统计标签分布情况，发现只要排除标签’0‘即可
    :return:
    '''
    path = r"E:\pycharm\github_reps\online_git\ttt\ruiying\data\all_wood_data_from_kaggle\Bounding Boxes - YOLO Format - 1\Bounding Boxes - YOLO Format - 1"
    path = r"E:\pycharm\github_reps\online_git\ttt\ruiying\data\new_data\labels\train"
    txt_files = read_all_txt_files(path)

    labels = ["0","1","2","3","4","5","6","7"]
    result = {key:0 for key in labels}
    print(result)

    for idx,file in enumerate(txt_files):
        print(idx,file)

        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                newline = line.split(" ")
                # print(newline)

                label_index = newline[0]

                result[label_index] = result[label_index] + 1

        # if idx > 2:
        #     break

    print(result)


    # 统计结果如下
    {'0': 171, '1': 4070, '2': 206, '3': 650, '4': 2934, '5': 542, '6': 121, '7': 517}
    # 对比论文，把‘0’排除即可，那么数据就能完全对上

# get_labels_st()


def analysis_labels_0():
    path = r"E:\pycharm\github_reps\online_git\ttt\ruiying\data\all_wood_data_from_kaggle\Bounding Boxes - YOLO Format - 1\Bounding Boxes - YOLO Format - 1"
    txt_files = read_all_txt_files(path)

    # labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    # result = {key: 0 for key in labels}
    # print(result)


    cnts = 0

    for idx, file in enumerate(txt_files):
        # print(idx, file)

        new_set = set()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                newline = line.split(" ")
                # print(newline)

                label_index = newline[0]

                new_set.add(label_index)

        if "0" in new_set and len(new_set) > 1:
            print(idx,file)
            cnts += 1
            # 说明包含多个标签，且有“0”的图像，111张
        elif len(new_set) == 0:
            # 说明正常图像，有388张，完全没有任何缺陷
            print("没有值")
            cnts += 1000


    print(cnts)

        # if idx > 2:
        #     break

    # print(result)



def move_labels_1():
    '''
    给所有的数据标签-1

    # 考虑使用百度网盘的数据 -- 优先  -- 数据和论文中的对应不上
    # 或者这里自行处理，删除标签为’0‘的，然后其他标签依次-1即可 -- 待处理

    :return:
    '''
    '''
    统计标签分布情况，发现只要排除标签’0‘即可
    :return:
    '''
    path = r"E:\pycharm\github_reps\online_git\ttt\ruiying\data\all_wood_data_from_kaggle\Bounding Boxes - YOLO Format - 1\Bounding Boxes - YOLO Format - 1"
    # path = r"E:\pycharm\github_reps\online_git\ttt\ruiying\data\new_data\labels\train"
    txt_files = read_all_txt_files(path)

    # gen_path = r'E:\pycharm\github_reps\online_git\ttt\ruiying\data\all_wood_data_from_kaggle\gen' # 保留Quartzity标签，只做循环移位
    gen_path = r'E:\pycharm\github_reps\online_git\ttt\ruiying\data\all_wood_data_from_kaggle\gen_without_Quartzity' # 不保留Quartzity标签

    labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    result = {key: 0 for key in labels}
    print(result)

    for idx, file in enumerate(txt_files):
        print(idx, file)

        # 创建输出文件路径
        # print("当前的filename",os.path.basename(file))
        output_file = os.path.join(gen_path, os.path.basename(file))
        # print(output_file)
        with open(output_file, 'w', encoding='utf-8') as fw:
            # fw.write(newline)

            collect_cnt = 0
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    newline = line.split(" ")
                    # print("读取的结果",newline)

                    label_index = int(newline[0])
                    if label_index == 0:
                        label_index = 7
                        continue
                    else:
                        label_index = label_index - 1
                        collect_cnt += 1
                    newline[0] = str(label_index)

                    newline = " ".join(newline)

                    # print("新的",newline)
                    # print(" ")
                    fw.write(newline) # 写入文件
            if collect_cnt == 0:
                fw.write("")  # 写入文件


        # if idx > 2:
        #     break

    # print(result)
    #
    # # 统计结果如下
    # {'0': 171, '1': 4070, '2': 206, '3': 650, '4': 2934, '5': 542, '6': 121, '7': 517}
    # # 对比论文，把‘0’排除即可，那么数据就能完全对上


move_labels_1()

















