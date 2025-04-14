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
from pydantic import BaseModel
import requests
import json
import ffmpeg
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
from ultralytics.engine.results import Boxes

class ObjectInfo(BaseModel):
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

class FrameInfo(BaseModel):
    frame_id: int
    objects: List[ObjectInfo]

class VideoInfo(BaseModel):
    length: int
    frames: List[FrameInfo]

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

model = YOLO("E:\PycharmProjects\pythonProject/runs/train\cleaned-data\model_l_size_736_wfs_with_aug2\weights/best.pt")
model1 = YOLO("yolov8x.pt")


@app.get("/fileRecognize", response_model=PicInfo)
async def fileRecognize(filePath: str):
    picInfo = predict_img(filePath)
    return picInfo


@app.get("/dirRecognize")
async def dir_recognize(dirPath: str, requestUrl: str, taskId: str ,modelName: str):
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


def img_count(filePath):
    im2 = cv2.imread(filePath)
    # 进行预测
    results = model1.predict(source=im2, conf=0.3)
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
        for root, dirs, files in os.walk(directory):
            for file in files:

                # 得到文件后缀名  需要根据情况进行修改
                suffix = file.split("/")[-1][file.split("/")[-1].index(".") + 1:]
                prefix = file.split("/")[-1][:file.split("/")[-1].index(".")]

                if suffix.lower() in ['png', 'jpg', 'jpeg']:
                    filePath = os.path.join(root, file.replace("\\", "/"))
                    im2 = cv2.imread(filePath)
                    # 进行预测
                    results = model.predict(source=im2, conf=0.3)

                    boxes = results[0].boxes.xyxy.tolist()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    class_names = [model.names[int(cls)] for cls in class_ids]
                    confs = results[0].boxes.conf.cpu().numpy().astype(float).tolist()

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

                    cap = cv2.VideoCapture(filePath)
                    class_set = predict_video(directory, prefix, cap)

                    file = file.replace(suffix, "mp4")

                    yolo_info = YoloInfo(classNames=list(class_set), boxes=[], confs=[])
                    pic_info = PicInfo(
                        picName=file,
                        oriPicPath=filePath,
                        firstPic=os.path.join(directory, 'cover', prefix + '.jpg'),
                        recPicPath=os.path.join(directory, 'label', file),
                        yoloInfo=yolo_info
                    )
                    list_info.append(pic_info)

    return list_info


def call_java_interface_for_error_handling(taskId):
    aiqxTaskAddVO = {
        "id": taskId,
        "status": 3
    }

    # 将aiqxTaskAddVO转换为JSON（如果需要的话，但在这个例子中Java接口可能接受JSON或表单数据）
    aiqxTaskAddVO_json = json.dumps(aiqxTaskAddVO)

    # 调用Java接口
    error_handling_url = "http://192.16.16.32:8101/aiqx/other/editTask"
    response = requests.post(error_handling_url, json=aiqxTaskAddVO)  # 或者使用data=aiqxTaskAddVO_json如果接口需要表单数据

    # 检查响应状态码，但在这个例子中我们可能只关心它是否成功发送
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


def predict_video(filePath, fileName, cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 10.0

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

    # FFmpeg 输出流
    process = (
        ffmpeg
        .input('pipe:0', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=fps)
        .output(video_path, pix_fmt='yuv420p', vcodec='libx264')
        .run_async(pipe_stdin=True)
    )

    class_set = set()  # 使用集合来避免物种名称重复

    # 初始化变量
    frames = []
    frame_id = 0

    # 获取视频总帧数
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 对帧运行 YOLO 推理
        results = model(frame, conf=0.3)
        result = results[0]
        boxes = result.boxes

        # 提取置信度、类别ID和边界框坐标
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()

        # 创建 ObjectInfo 列表
        objects = []
        for class_id, conf, (x1, y1, x2, y2) in zip(class_ids, confs, xyxys):
            object_info = ObjectInfo(
                class_id=int(class_id),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf)
            )
            objects.append(object_info)

        # 创建 FrameInfo
        frame_info = FrameInfo(
            frame_id=frame_id,
            objects=objects
        )

        # 将 FrameInfo 添加到 frames 列表
        frames.append(frame_info)

        # 累加帧 ID
        frame_id += 1

    # 释放资源
    cap.release()
    process.stdin.close()
    process.wait()

    # 创建 VideoInfo
    video_info = VideoInfo(
        length=video_length,
        frames=frames
    )
    with open('video_info.json', 'w') as json_file:
        json.dump(video_info.dict(), json_file, indent=4)




def predict_img(fileName) -> PicInfo:
    # 分割路径和文件名
    directory, file_name = os.path.split(fileName)

    im2 = cv2.imread(fileName)
    results = model.predict(source=im2, conf=0.3)  # 将预测保存为标签

    boxes = results[0].boxes.xyxy.tolist()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[int(cls)] for cls in class_ids]
    confs = results[0].boxes.conf.cpu().numpy().astype(float).tolist()

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
    cap = cv2.VideoCapture('E:/download/1.mp4')
    predict_video('E:/download', 'iomoreimgcom', cap)