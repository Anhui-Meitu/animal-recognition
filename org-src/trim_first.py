import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm  # 用于显示进度条，需要安装

def trim_first_second(input_folder, output_folder=None):
    """
    截取文件夹内所有视频的第一秒内容
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径（默认为输入文件夹+"_trimmed"）
    """
    # 设置输出路径
    output_folder = output_folder or f"{input_folder}_trimmed"
    os.makedirs(output_folder, exist_ok=True)

    # 支持的视频格式
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # 获取视频文件列表
    video_files = [f for f in os.listdir(input_folder)
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"发现 {len(video_files)} 个视频文件需要处理")

    for filename in tqdm(video_files, desc="处理进度"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 读取视频文件
            with VideoFileClip(input_path) as video:
                # 检查视频时长
                if video.duration <= 1:
                    print(f"\n跳过 {filename}（时长不足1秒）")
                    continue

                # 截取从1秒到结束的内容
                trimmed = video.subclip(3, video.end)

                # 写入输出文件（保持原视频参数）
                trimmed.write_videofile(
                    output_path,
                    codec='libx264',  # H.264编码
                    audio_codec='aac',  # AAC音频编码
                    preset='medium',   # 编码速度与质量的平衡
                    threads=4,         # 使用多线程加速
                    logger=None        # 关闭进度输出
                )

        except Exception as e:
            print(f"\n处理 {filename} 时出错：{str(e)}")
            continue

if __name__ == "__main__":
    # 使用示例
    input_folder = input("请输入要处理的文件夹路径：").strip()
    trim_first_second(input_folder)