from inspect import FrameInfo

import torch
from fastapi import FastAPI, HTTPException
# 导入程序运行必须模块
import sys
import time

import cv2
import os
from ultralytics import YOLO
from typing import List
from pydantic import BaseModel # for data validation and conversions
import requests
import json
import ffmpeg
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
from ultralytics.engine.results import Boxes
import subprocess


class YoloInfo(BaseModel):
    classNames: List[str]
    boxes: List[List[float]]
    confs: List[float]


class PicInfo(BaseModel):
    picName: str
    oriPicPath: str
    recPicPath: str
    firstPic: str
    yoloInfo: YoloInfo


class AIVO(BaseModel):
    aiResult: str


app = FastAPI()

model = YOLO("tianma.pt")
model1 = YOLO("yolov8x.pt")
model2 = YOLO("fuhe.pt")


@app.get("/fileRecognize", response_model=PicInfo)
async def fileRecognize(filePath: str):
    picInfo = predict_img(filePath)
    return picInfo


@app.get("/dirRecognize")
async def dir_recognize(dirPath: str, requestUrl: str, taskId: str, modelName: str):
    # 检查 modelName 是否不为空
    global model  # 声明使用全局变量
    if modelName:
        model = YOLO(modelName)  # 使用指定的模型
    else:
        model = YOLO("tianma.pt")  # 保持默认模型

    list1 = input_line5(dirPath)
    try:
        # 将list1转换为JSON字符串
        aiResult = json.dumps([item.dict() for item in list1])

        print(aiResult)

        # 要发送的数据
        data = {
            "aiResult": aiResult
        }

        # 调用Java接口
        response = requests.post(requestUrl, json=data)
        response.raise_for_status()

        # 返回Java接口的响应
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        call_java_interface_for_error_handling(taskId)
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except Exception as err:
        call_java_interface_for_error_handling(taskId)
        raise HTTPException(status_code=500, detail=str(err))


def voice_recognize(folder_path, task_id):
    results = []
    # 构建 json 数据
    json_data = {
        "output_threshold": 0.05,
        "num_results": 1
    }
    # 将 json 数据转换为符合规范的 JSON 字符串
    json_str = json.dumps(json_data)
    # 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # 读取文件
            with open(file_path, 'rb') as file:
                # 构建请求参数
                files = {
                    'audio': file
                }
                data = {
                    'meta': json_str
                }
                try:
                    # 发送 POST 请求
                    response = requests.post('http://192.16.16.182:8080/analyze', files=files, data=data)

                    # 处理响应
                    if response.status_code == 200:
                        response_body = response.text
                        response_json = json.loads(response_body)
                        result = {
                            'task_id': task_id,
                            'file_name': file_name,
                            'response_body': response_json
                        }
                        results.append(result)
                    else:
                        result = {
                            'task_id': task_id,
                            'file_name': file_name,
                            'error': f"请求失败，状态码: {response.status_code}"
                        }
                        results.append(result)
                except requests.RequestException as e:
                    result = {
                        'task_id': task_id,
                        'file_name': file_name,
                        'error': f"请求发生错误: {e}"
                    }
                    results.append(result)
                except Exception as e:
                    result = {
                        'task_id': task_id,
                        'file_name': file_name,
                        'error': f"发生未知错误: {e}"
                    }
                    results.append(result)

    return results


@app.get("/voiceDirRecognize")
async def voice_dir_recognize(dirPath: str, requestUrl: str, taskId: str):
    print(dirPath)

    list1 = voice_recognize(dirPath, taskId)
    try:
        # 将result转换为JSON字符串
        aiResult = json.dumps(list1)

        print(aiResult)
        # 要发送的数据
        data = {
            "aiResult": aiResult
        }

        # 调用Java接口
        response = requests.post(requestUrl, json=data)
        response.raise_for_status()

        # 返回Java接口的响应
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        call_voice_update_state(taskId)
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except Exception as err:
        call_voice_update_state(taskId)
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/videoRecognize")
async def videoRecognize(videoPath: str):
    input_line2(videoPath)
    return {"message": videoPath}


@app.get("/streamRecognize")
async def streamRecognize(streamPath: str):
    stream_rec(streamPath)
    return {"message": streamPath}


@app.get("/fileCount")
async def fileCount(filePath: str):
    # 得到文件后缀名  需要根据情况进行修改
    suffix = filePath.split("/")[-1][filePath.split("/")[-1].index(".") + 1:]
    count_num = 0
    if suffix.lower() in ['png', 'jpg', 'jpeg']:
        count_num = img_count(filePath)
    elif suffix.lower() in ['mp4', 'avi', 'wmv', 'mpeg']:
        count_num = video_count(filePath)
    return {"count": count_num}


@app.get("/updateRecName")
async def updateRecName(filePath: str, updateName: str, fileId: str):
    video_path = updateRecName(filePath, updateName)
    print(video_path)
    # 调用Java接口更新文件路径
    try:
        response = call_java_interface_update_path(fileId, video_path)
        print(response)
    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))


def updateRecName(filePath, updateName):
    # 分割文件路径和文件名
    file_dir, file_fullname = os.path.split(filePath)
    file_name = os.path.splitext(file_fullname)[0]  # 去除扩展名

    # 创建输出目录结构
    output_video_dir = os.path.join(file_dir, 'edit')
    output_cover_dir = os.path.join(file_dir, 'cover')
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_cover_dir, exist_ok=True)

    # 初始化视频流
    cap = cv2.VideoCapture(filePath)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    # 确保分辨率为偶数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 修改FFmpeg命令（显式指定输入/输出格式）
    video_path = os.path.join(output_video_dir, f'{file_name}.mp4')
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"视频文件 {video_path} 已成功删除。")
        except Exception as e:
            print(f"删除视频文件 {video_path} 时出错: {e}")

    video_pic_path = os.path.join(output_cover_dir, f'{file_name}.jpg')
    # 修正后的FFmpeg参数（移除冲突的r=orig_fps）
    # FFmpeg 输出流
    process = (
        ffmpeg
        .input('pipe:0', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=orig_fps)
        .output(video_path, pix_fmt='yuv420p', vcodec='libx264')
        .run_async(pipe_stdin=True)
    )

    # 获取中文类别到ID的映射
    def get_class_id(name: str) -> int:
        for idx, cn_name in enumerate(model2.names.values()):
            if cn_name == name:
                print(cn_name)
                return idx
        raise ValueError(f"类别 '{name}' 不存在于模型")

    target_class_id = get_class_id(updateName)
    no_continuous_frame = 0

    cover_flag = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

            # 调整帧大小至偶数
        frame = cv2.resize(frame, (width, height))

        if not cover_flag:
            # 先保存第一帧 后面如果识别到物种再替换
            cv2.imwrite(video_pic_path, frame)

        # 对帧运行 YOLOv8 推理
        results = model(frame, conf=0.3, imgsz=736)
        results1 = model2(frame, conf=0.3, imgsz=736)
        torch.cuda.empty_cache()  # Clear unused GPU memory

        # 获取 model 识别的边界框信息
        xyxy_model = results[0].boxes.xyxy
        conf_model = results[0].boxes.conf

        # 如果 model2 没有识别出目标，使用默认的 classid
        new_cls = torch.tensor([target_class_id] * xyxy_model.shape[0]).to(xyxy_model.device)

        conf_model = conf_model.unsqueeze(1)
        new_cls = new_cls.unsqueeze(1)

        # 拼接新的边界框数据
        new_boxes_data = torch.cat([xyxy_model, conf_model, new_cls], dim=1)
        orig_shape = results[0].boxes.orig_shape

        # 创建全新的 Boxes 对象
        new_boxes = Boxes(new_boxes_data, orig_shape=orig_shape)

        # 创建一个新的空结果对象，避免混入 model2 原本的框
        from copy import deepcopy
        new_result = deepcopy(results1[0])
        new_result.boxes = new_boxes

        # 绘制标注框
        annotated_frame = new_result.plot()

        if not cover_flag:
            # 保存第一帧识别图片做封面
            cv2.imwrite(video_pic_path, annotated_frame)
            cover_flag = True

        if annotated_frame is not None:
            # 写入视频帧
            if process.poll() is None:  # FFmpeg 进程仍在运行
                process.stdin.write(annotated_frame.astype(np.uint8).tobytes())
            else:
                print("FFmpeg进程不存在")

    compress_image(video_pic_path, video_pic_path)
    # 释放资源
    cap.release()
    if process:
        process.stdin.close()
        process.wait()

    return video_path


def compress_image(input_path, output_path, quality=50):
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 保存压缩后的图片
            img.save(output_path, optimize=True, quality=quality)
            print(f"图片已成功压缩并保存到 {output_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_path}")
    except Exception as e:
        print(f"发生未知错误：{e}")


def img_count(filePath):
    im2 = cv2.imread(filePath)
    # 进行预测
    results = model1.predict(source=im2, conf=0.3)
    torch.cuda.empty_cache()  # Clear unused GPU memory
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    count_num = class_ids.size()
    return count_num


def video_count(filePath):
    total_detected = 0
    cap = cv2.VideoCapture(filePath)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 使用os.path.dirname获取文件所在的目录
    dir_path = os.path.dirname(filePath)
    # 使用os.path.basename获取文件名
    file_name = os.path.basename(filePath)
    # 替换文件名
    new_file_name = file_name.replace(".mp4", "_rec.mp4") if ".mp4" in file_name else file_name + "_rec.mp4"
    # 重新拼接路径
    new_path = os.path.join(dir_path, new_file_name)
    video_writer = cv2.VideoWriter(new_path, fourcc, fps, (frame_width, frame_height))

    track_history = {}
    seen_ids = set()
    track_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model1(frame)
        torch.cuda.empty_cache()  # Clear unused GPU memory
        boxes = results[0].boxes

        new_track_history = {}

        if boxes is not None:
            boxes = boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Find the closest existing ID or create a new one
                min_distance = float('inf')
                min_id = None
                for tid, center in track_history.items():
                    distance = np.linalg.norm(np.array(bbox_center) - np.array(center))
                    if distance < min_distance:
                        min_distance = distance
                        min_id = tid

                if min_distance < 50:  # Threshold to assign the ID to the closest existing box
                    new_track_history[min_id] = bbox_center
                    current_id = min_id
                else:
                    new_track_history[track_id] = bbox_center
                    current_id = track_id
                    track_id += 1

                # If current ID is new, add it to seen_ids and increment the total count
                if current_id not in seen_ids:
                    seen_ids.add(current_id)
                    total_detected += 1

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, bbox_center, 2, (0, 0, 255), -1)
                cv2.putText(frame, f'ID: {current_id}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the track history with the new IDs for the next frame
        track_history = new_track_history

        # Show total detected objects in the left upper corner
        cv2.putText(frame, f'Total Detected: {total_detected}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video_writer.write(frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    return total_detected


def stream_rec(streamPath):
    cap = cv2.VideoCapture(streamPath)
    # 存储视频路径
    video_path = '/data/video'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    # video_path = os.path.join(video_path, fileName + '.mp4')

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    out = None
    recording = False
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform YOLO detection
        results = model(frame)
        torch.cuda.empty_cache()  # Clear unused GPU memory
        result = results[0]

        # Check if any objects are detected
        if len(result) > 0:
            annotated_frame = result.plot()

            if not recording:
                # Start a new video file
                video_filename = os.path.join('/data/video', f'detected_{int(time.time())}.mp4')
                out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'xmp4'), frame_rate,
                                      (frame_width, frame_height))
                recording = True
                print(f"Recording started: {video_filename}")

            # Write the frame to the video file
            out.write(annotated_frame)

            # Display the frame with detections
            cv2.imshow('YOLO Stream Detection', annotated_frame)
        else:
            if recording:
                # Stop recording
                recording = False
                out.release()
                print("Recording stopped and file saved.")

    # Release the capture and close the window
    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()


def input_line5(directory) -> List[PicInfo]:
    list_info = []

    if directory:
        # for root, dirs, files in os.walk(directory):
        files = os.listdir(directory)
        root = directory
        for file in files:
            if not os.path.isfile(os.path.join(root, file)): continue

            # 得到文件后缀名  需要根据情况进行修改
            # suffix = file.split("/")[-1][file.split("/")[-1].index(".") + 1:]
            # prefix = file.split("/")[-1][:file.split("/")[-1].index(".")]
            suffix = file.split('.')[-1]
            prefix = '.'.join(file.split('.')[:-1])

            if suffix.lower() in ['png', 'jpg', 'jpeg']:
                filePath = os.path.join(root, file.replace("\\", "/"))

                im2 = cv2.imread(filePath)
                # 进行预测
                results = model.predict(source=im2, conf=0.3)

                boxes = results[0].boxes.xyxy.tolist()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[int(cls)] for cls in class_ids]
                confs = results[0].boxes.conf.cpu().numpy().astype(float).tolist()
                
                torch.cuda.empty_cache()  # Clear unused GPU memory

                # 获取带有标注的图片
                annotated_frame = results[0].plot()

                # 根据是否有预测结果保存图片
                if len(results[0].boxes) > 0:
                    output_folder = os.path.join(directory, 'label')
                else:
                    output_folder = os.path.join(directory, 'no_label')

                # 判断文件夹是否存在 不存在则创建
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # 保存带有标注的图片
                output_image_file = os.path.join(output_folder, file)
                cv2.imwrite(output_image_file, annotated_frame)

                yolo_info = YoloInfo(classNames=class_names, boxes=boxes, confs=confs)
                pic_info = PicInfo(
                    picName=file,
                    oriPicPath=filePath,
                    firstPic='',
                    recPicPath=output_image_file,
                    yoloInfo=yolo_info
                )
                list_info.append(pic_info)

            elif suffix.lower() in ['mp4', 'avi', 'wmv', 'mpeg']:
                filePath = os.path.join(root, file.replace("\\", "/"))

                # 预测视频
                cap = cv2.VideoCapture(filePath)
                class_set = predict_video(directory, prefix, cap, confidence=0.6, continuous_frame_threshold=1)

                # TODO: conversion needed? 需不需要转换格式
                file = file.replace(suffix, "mp4")

                # 构造路径
                rec_pic_path = os.path.join(directory, 'label', file)

                if not class_set:
                    recPicPath = ""
                else:
                    recPicPath = rec_pic_path

                yolo_info = YoloInfo(classNames=[] if not class_set else [class_set], boxes=[], confs=[])
                pic_info = PicInfo(
                    picName=file,
                    oriPicPath=filePath,
                    firstPic=os.path.join(directory, 'cover', prefix + '.jpg'),
                    recPicPath=recPicPath,
                    yoloInfo=yolo_info
                )
                list_info.append(pic_info)
                
                cap.release() # ensure resources are freed

    return list_info


def call_java_interface_for_error_handling(taskId):
    aiqxTaskAddVO = {
        "id": taskId,
        "status": 3
    }

    # 将aiqxTaskAddVO转换为JSON（如果需要的话，但在这个例子中Java接口可能接受JSON或表单数据）
    aiqxTaskAddVO_json = json.dumps(aiqxTaskAddVO)

    # 调用Java接口
    error_handling_url = "http://192.16.16.182:8101/aiqx/other/editTask"
    response = requests.post(error_handling_url, json=aiqxTaskAddVO)  # 或者使用data=aiqxTaskAddVO_json如果接口需要表单数据

    # 检查响应状态码
    response.raise_for_status()  # 这将再次抛出HTTPError如果状态码不是2xx

    # 返回响应对象（或根据需要处理它）
    return response.json()  # 或者只是return response如果你不需要JSON内容


def call_voice_update_state(taskId):
    aiqxTaskAddVO = {
        "id": taskId,
        "taskState": 3
    }

    # 将aiqxTaskAddVO转换为JSON（如果需要的话，但在这个例子中Java接口可能接受JSON或表单数据）
    aiqxTaskAddVO_json = json.dumps(aiqxTaskAddVO)

    # 调用Java接口
    error_handling_url = "http://192.16.16.182:8101/aiqx/voice/editVoiceTask"
    response = requests.post(error_handling_url, json=aiqxTaskAddVO)  # 或者使用data=aiqxTaskAddVO_json如果接口需要表单数据

    # 检查响应状态码
    response.raise_for_status()  # 这将再次抛出HTTPError如果状态码不是2xx

    # 返回响应对象（或根据需要处理它）
    return response.json()  # 或者只是return response如果你不需要JSON内容


def call_java_interface_update_path(fileId, updatePath):
    aiqxTaskAddVO = {
        "id": fileId,
        "path": updatePath
    }

    # 将aiqxTaskAddVO转换为JSON（如果需要的话，但在这个例子中Java接口可能接受JSON或表单数据）
    aiqxTaskAddVO_json = json.dumps(aiqxTaskAddVO)

    # 调用Java接口
    error_handling_url = "http://192.16.16.182:8101/aiqx/aiqxFile/updateFile"
    response = requests.post(error_handling_url, json=aiqxTaskAddVO)  # 或者使用data=aiqxTaskAddVO_json如果接口需要表单数据

    # 检查响应状态码
    response.raise_for_status()  # 这将再次抛出HTTPError如果状态码不是2xx

    # 返回响应对象（或根据需要处理它）
    return response.json()  # 或者只是return response如果你不需要JSON内容


def input_line2(file_name):
    # 得到文件后缀名  需要根据情况进行修改
    suffix = file_name.split("/")[-1][file_name.split("/")[-1].index(".") + 1:]
    prefix = file_name.split("/")[-1][:file_name.split("/")[-1].index(".")]

    filePath = os.path.dirname(file_name)

    if file_name == '':
        pass
    elif suffix.lower() in ['mp4', 'avi', 'wmv', 'mpeg']:
        cap = cv2.VideoCapture(file_name)
    predict_video(filePath, prefix, cap)
    


def predict_video(filePath, fileName, cap, confidence=0.3, continuous_frame_threshold=1):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频存储路径
    video_path = os.path.join(filePath, 'label')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    video_path = os.path.join(video_path, fileName + '.mp4')

    # 创建封面图片路径
    video_pic_path = os.path.join(filePath, 'cover')
    if not os.path.exists(video_pic_path):
        os.makedirs(video_pic_path)
    video_pic_path = os.path.join(video_pic_path, fileName + '.jpg')

    class_counter = {}  # 记录每个类别出现的次数

    # 初始化标志和状态变量
    first_frame_saved = False

    # 用于记录连续未检测到目标的帧数
    continuous_frame = 0
    no_continuous_frame = 0
    skip_second_pass = True

    # 第一遍遍历：记录类别出现次数
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 如果第一帧没有存储
        if not first_frame_saved:
            cv2.imwrite(video_pic_path, frame)  # 保存第一帧为图片
            first_frame_saved = True

        # 对帧运行 YOLOv8 推理
        results = model(frame, conf=confidence)
        torch.cuda.empty_cache()  # Clear unused GPU memory
        
        result = results[0]
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy()  # 提取类别ID

        if len(class_ids) > 0:
            continuous_frame += 1
            if continuous_frame >= continuous_frame_threshold:
                skip_second_pass = False
            # 计数类别出现次数
            for cls_id in class_ids:
                class_counter[cls_id] = class_counter.get(cls_id, 0) + 1
        else:
            continuous_frame = 0  # 重置连续检测到帧数

    # 如果连续N帧未检测到，不执行第二遍遍历
    if skip_second_pass:
        cap.release()
        return None

    # FFmpeg 输出流
    process = (
        ffmpeg
        .input('pipe:0', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=fps)
        .output(video_path, pix_fmt='yuv420p', vcodec='libx264')
        .run_async(pipe_stdin=True)
    )

    # 根据类别出现次数确定最多的类别
    most_common_class = max(class_counter, key=class_counter.get)
    print('most common class: ', most_common_class)

    # 重置视频读取指针
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_boxes_data = None
    # 第二遍遍历：根据检测结果绘制并处理未检测到目标的帧
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = None

        # 对帧运行 YOLOv8 推理
        results = model(frame, conf=confidence, imgsz=736)
        torch.cuda.empty_cache()  # Clear unused GPU memory
        result = results[0]
        boxes = result.boxes
        class_ids = results[0].boxes.cls.cpu().numpy()

        # 保存当前帧的边界框信息
        last_boxes_data = (boxes.xyxy, boxes.conf, class_ids)

        if boxes.shape[0] > 0:

            no_continuous_frame = 0

            # 保存当前帧的边界框信息
            xyxy = boxes.xyxy
            conf = boxes.conf
            new_cls = torch.tensor([most_common_class] * len(xyxy)).to(boxes.cls.device)  # 使用最多的类别

            # 确保维度一致
            if xyxy.shape[0] == conf.shape[0] == new_cls.shape[0]:
                conf = conf.unsqueeze(1)
                new_cls = new_cls.unsqueeze(1)
                new_boxes_data = torch.cat([xyxy, conf, new_cls], dim=1)
                orig_shape = boxes.orig_shape

                new_boxes = Boxes(new_boxes_data, orig_shape=orig_shape)
                result.boxes = new_boxes
                annotated_frame = result.plot()

        else:
            no_continuous_frame += 1
            if no_continuous_frame < 15:
                if last_boxes_data is not None:
                    xyxy, conf, class_ids = last_boxes_data
                    new_cls = torch.tensor([most_common_class] * len(xyxy)).to(boxes.cls.device)

                    if xyxy.shape[0] == conf.shape[0] == new_cls.shape[0]:
                        conf = conf.unsqueeze(1)
                        new_cls = new_cls.unsqueeze(1)
                        new_boxes_data = torch.cat([xyxy, conf, new_cls], dim=1)
                        orig_shape = boxes.orig_shape

                        new_boxes = Boxes(new_boxes_data, orig_shape=orig_shape)
                        result.boxes = new_boxes
                        annotated_frame = result.plot()
                else:
                    annotated_frame = frame
            else:
                annotated_frame = frame

        if annotated_frame is not None:
            # 写入视频帧
            if process.poll() is None:  # FFmpeg 进程仍在运行
                process.stdin.write(annotated_frame.astype(np.uint8).tobytes())
            else:
                print("FFmpeg进程不存在")

    # 释放资源
    cap.release()
    if process:
        process.stdin.close()
        process.wait()

    return model.names[int(most_common_class)]


def predict_img(fileName) -> PicInfo:
    # 分割路径和文件名
    directory, file_name = os.path.split(fileName)

    im2 = cv2.imread(fileName)
    results = model.predict(source=im2, conf=0.3, imagsz=736)  # 将预测保存为标签

    boxes = results[0].boxes.xyxy.tolist()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[int(cls)] for cls in class_ids]
    confs = results[0].boxes.conf.cpu().numpy().astype(float).tolist()
    
    torch.cuda.empty_cache()  # Clear unused GPU memory

    # 获取带有标注的图片
    annotated_frame = results[0].plot()

    # 根据是否有预测结果保存图片
    if len(results[0].boxes) > 0:
        output_folder = os.path.join(directory, 'label')
    else:
        output_folder = os.path.join(directory, 'no_label')

    # 判断文件夹是否存在 不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存带有标注的图片
    output_image_file = os.path.join(output_folder, file_name)
    cv2.imwrite(output_image_file, annotated_frame)

    yolo_info = YoloInfo(classNames=class_names, boxes=boxes, confs=confs)
    pic_info = PicInfo(
        picName=file_name,
        oriPicPath=fileName,
        firstPic='',
        recPicPath=output_image_file,
        yoloInfo=yolo_info
    )

    return pic_info


if __name__ == '__main__':
    # 本地测试
    folder_path = 'C:/Users/17756\Desktop/fsdownload/voice'
    all_results = voice_recognize(folder_path, 1)
    # 将result转换为JSON字符串
    aiResult = json.dumps(all_results)

    print(aiResult)
    print(all_results)
