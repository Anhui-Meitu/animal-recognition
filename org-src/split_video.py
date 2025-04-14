import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    """
    从视频中按每秒切割帧并保存为图片。
    
    参数：
    - video_path: 输入视频的路径。
    - output_dir: 输出图片的目录。
    - fps: 每秒提取的帧数（默认 1）。
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return
    
    # 获取视频的帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(video_fps)
    if video_fps == 0:
        print("无法获取视频帧率，确保视频文件正确。")
        return

    interval = max(int(video_fps // fps), 1)  # 计算提取帧的间隔，确保至少为 1

    print(interval)
    
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔 interval 帧保存一张图片
        # if frame_count % interval == 0:
           
        # frame_count += 1
        output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg").replace("\\", "/")
        success = cv2.imwrite(output_path, frame)
        if success:
            print(f"保存图片：{output_path}")
            saved_count += 1
        else:
            print(f"保存失败：{output_path}")
        
    
    cap.release()
    print(f"提取完成，总共保存了 {saved_count} 张图片。")


# 示例调用
extract_frames("F:/ahs.MP4", "C:/Users/LL/Desktop/pic/3")
