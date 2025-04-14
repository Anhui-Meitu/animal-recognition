import json
import os

#将anylabeling json格式转换为yolo格式数据

def labelme_to_yolo(label_me_json_file, cls2id_dict):
    label_me_json = json.load(open(label_me_json_file, mode='r', encoding='utf-8'))
    shapes = label_me_json['shapes']
    img_width, img_height = label_me_json['imageWidth'], label_me_json['imageHeight']
    img_path = label_me_json['imagePath']
    img_data = label_me_json['imageData'] if 'imageData' in label_me_json else ''

    labels = []
    for s in shapes:
        s_type = s['shape_type']
        s_type = s_type.lower()
        if s_type == 'rectangle':
            pts = s['points']
            x1, y1 = pts[0]  # left corner
            x2, y1 = pts[1]  # right corner
            x2, y2 = pts[2]  # right corner
            x1, y2 = pts[3]  # right corner
            x = (x1 + x2) / 2 / img_width
            y = (y1 + y2) / 2 / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height
            if s['label'] in cls2id_dict:
                cid = cls2id_dict[s['label']]
            else:
                cid = 0
            labels.append(f'{cid} {x} {y} {w} {h}')

    return labels


def write_label2txt(save_txt_path, label_list):
    f = open(save_txt_path, "w", encoding="UTF-8")

    for label in label_list:
        temp_list = label.split(" ")
        f.write(temp_list[0])
        f.write(" ")
        f.write(temp_list[1])
        f.write(" ")
        f.write(temp_list[2])
        f.write(" ")
        f.write(temp_list[3])
        f.write(" ")
        f.write(temp_list[4])
        f.write("\n")


if __name__ == '__main__':
    # 原始图片文件夹路径
    # img_dir = r"\\192.16.16.2\项目备份\红外相机数据\天马保护区\20240618\小麂\ces"
    # 原始JSON标签文件夹路径
    json_dir = r"\\192.16.16.2\项目备份\红外相机数据\01、标注成果-胡媛媛\虎斑地鸫\json"
    # 生成保存TXT文件夹路径
    save_dir = r"\\192.16.16.2\项目备份\红外相机数据\01、标注成果-胡媛媛\虎斑地鸫\yolo"
    # 类别和序号的映射字典
    cls2id_dict = {"虎斑地鸫": "17"}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for json_name in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_name)
        txt_name = json_name.split(".")[0] + ".txt"
        save_txt_path = os.path.join(save_dir, txt_name)
        labels = labelme_to_yolo(json_path, cls2id_dict)
        write_label2txt(save_txt_path, labels)