import os

def get_txt_file_names(txt_folder):
    """
    获取文件夹中的所有txt文件名（不带扩展名）
    """
    txt_files = [f.replace('.txt', '') for f in os.listdir(txt_folder) if f.endswith('.txt')]
    return set(txt_files)

def get_image_file_names(image_folder):
    """
    获取文件夹中的所有图片文件名（不带扩展名）
    """
    image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    return set(image_files)

def delete_unmatched_images(image_folder, matched_files):
    """
    删除文件夹中没有匹配到txt文件的图片
    """
    image_files = os.listdir(image_folder)
    for image in image_files:
        image_name, ext = os.path.splitext(image)
        if image_name not in matched_files:
            image_path = os.path.join(image_folder, image)
            print(f"Deleting {image_path}")
            os.remove(image_path)

def main(txt_folder, image_folder):
    """
    主函数，执行文件匹配并删除未匹配的图片
    """
    # 获取txt文件中的名称
    matched_files = get_txt_file_names(txt_folder)
    
    # 获取图片文件中的名称
    image_files = get_image_file_names(image_folder)
    
    # 找到匹配的文件
    matched_files_in_images = matched_files.intersection(image_files)
    
    # 删除未匹配的图片
    delete_unmatched_images(image_folder, matched_files_in_images)

if __name__ == '__main__':
    txt_folder = 'F:\池州参考训练数据\pic\label2'  # 替换为.txt文件所在文件夹路径
    image_folder = 'F:\池州参考训练数据\pic/2'  # 替换为图片所在文件夹路径
    
    main(txt_folder, image_folder)
