import os

def find_txt_files_with_first_char(directory, target_char):
    # 获取文件夹中的所有文件
    files = os.listdir(directory)
    
    # 用于存储符合条件的文件
    matching_files = []
    
    # 遍历文件夹中的每个文件
    for file in files:
        # 检查文件是否为txt文件
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as f:
                first_char = f.read(1)  # 读取第一个字符
                
                # 检查第一个字符是否为目标字符
                if first_char == target_char:
                    matching_files.append(file)
    
    return matching_files

# 设置要查找的文件夹路径
directory = 'E:/PycharmProjects/pythonProject/dataset/wfs_new/labels/yz/train'  # 替换为你的文件夹路径

# 设置目标字符
target_char = '1'

# 查找符合条件的文件
matching_files = find_txt_files_with_first_char(directory, target_char)

# 输出结果
if matching_files:
    print(f"以下文件的首个字符为 '{target_char}':")
    for file in matching_files:
        print(file)
else:
    print(f"没有找到首个字符为 '{target_char}' 的txt文件。")