import os
import shutil

def merge_folders_with_prefix(source_folders, target_folder):
    """
    合并多个文件夹中的文件，将文件名前加上所在文件夹的前缀
    :param source_folders: 包含源文件夹路径的列表
    :param target_folder: 目标文件夹路径
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for folder in source_folders:
        # 获取文件夹的名称，作为前缀
        folder_name = os.path.basename(folder)
        
        # 遍历文件夹中的文件
        for filename in os.listdir(folder):
            # 获取文件的完整路径
            file_path = os.path.join(folder, filename)
            
            # 检查是否是文件
            if os.path.isfile(file_path):
                # 创建新的文件名，添加前缀
                new_filename = f"{folder_name}_{filename}"
                new_file_path = os.path.join(target_folder, new_filename)
                
                # 复制文件到目标文件夹，并重命名
                shutil.copy(file_path, new_file_path)
                print(f"Copied {file_path} to {new_file_path}")

def main():
    source_folders = ['F:\池州参考训练数据\pic/1', 'F:\池州参考训练数据\pic/2']  # 替换为你的源文件夹路径
    target_folder = 'F:\池州参考训练数据/pic/train'  # 替换为目标文件夹路径
    merge_folders_with_prefix(source_folders, target_folder)

if __name__ == '__main__':
    main()
