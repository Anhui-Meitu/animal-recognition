def convert_to_yolo_format(input_file, output_file, image_width, image_height):
    """
    将数据转换为YOLO格式并保存到文件。
    :param input_file: 输入的文本文件路径。
    :param output_file: 输出的YOLO格式文件路径。
    :param image_width: 图像宽度。
    :param image_height: 图像高度。
    """
    yolo_lines = []

    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        # 提取每行数据
        x_min, y_min, width, height, *_ = map(int, line.strip().split(','))
        class_id = 0  # 默认类别ID（可以根据需要修改）

        # 计算中心点和宽高的归一化值
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        # 格式化为YOLO格式字符串
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    # 保存为YOLO格式文件
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_lines))
    print(f"转换完成，数据已保存到 {output_file}")

# 示例用法
input_file = "input.txt"  # 包含你提供数据的文件
output_file = "output_yolo.txt"  # 转换后的YOLO格式文件
image_width = 640  # 图像宽度（根据实际情况修改）
image_height = 480  # 图像高度（根据实际情况修改）

convert_to_yolo_format(input_file, output_file, image_width, image_height)
