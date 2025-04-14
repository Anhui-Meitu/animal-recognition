import os
import shutil
import random

def split_files(source_dir):
    """
    将文件夹中的文件按 7:2:1 分为三部分，
    分别创建到同级目录中的 val 和 test 文件夹。

    :param source_dir: 目标文件夹路径
    """
    # 检查目标文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"目录 {source_dir} 不存在！")
        return

    # 获取目标文件夹同级目录路径
    parent_dir = os.path.dirname(source_dir)


    # 定义 val 和 test 文件夹路径
    val_dir = os.path.join(parent_dir, f"val_hard_{source_dir.split("_")[-1]}")
    # test_dir = os.path.join(parent_dir, "test")

    # 如果 val 和 test 文件夹不存在则创建
    os.makedirs(val_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)

    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # # 随机打乱文件顺序
    # random.shuffle(files)

    # 计算划分数量
    total_files = len(files)
    val_count = total_files  // 10
    # test_count = total_files // 10

    # 分配文件
    val_files = files[:val_count]
    # test_files = files[val_count:val_count + test_count]

    # 将文件移动到对应文件夹
    for file in val_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))

    # for file in test_files:
    #     shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

    print(f"文件划分完成：")
    print(f"- {len(val_files)} 个文件移动到 {val_dir}")
    # print(f"- {len(test_files)} 个文件移动到 {test_dir}")

# 示例使用
split_files("E:\PycharmProjects\pythonProject\dataset/animal\images/train_hard_yz")
split_files("E:\PycharmProjects\pythonProject\dataset/animal\labels/train_hard_yz")