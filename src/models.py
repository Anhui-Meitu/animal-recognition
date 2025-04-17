"""
list of available models
"""
import os

import cv2
import torch
import ultralytics

def get_yolov11():
    """Returns a configured YOLOv11 model instance for training
        and inference
    """