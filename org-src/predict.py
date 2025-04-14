from ultralytics import YOLO
import cv2
import os

model = YOLO("E:\PycharmProjects\pythonProject\\runs\\train\cleaned-data\model_l_size_736_wfs_with_aug2\weights\\best.pt")
# model = YOLO("E:\PycharmProjects\pythonProject\wanfoshan.pt")
for file in os.listdir("G:\\test\\wfs\\"):
    model.predict(source="G:\\test\\wfs\\" + file, save=True, imgsz=736)
