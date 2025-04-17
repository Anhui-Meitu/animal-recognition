"""
Trains a model using the specified configuration file.
用于训练模型的脚本。
Designed to run from the command line.
设计为从命令行运行。
"""

import os
import sys
import argparse

import torch
from ultralytics import YOLO
import detectron2


def train_yolo(config_fname: str, model: str):
    model = YOLO("yolo11l.pt")
    model.train(
        data = '/mnt/e/wfs-dataset/data.yaml',
        epochs = 200,
        imgsz = 736,
        batch = 16,
        device = 0,
        save_dir = '/mnt/e/wfs-dataset/weights',
    )