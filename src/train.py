"""
Trains a model using the specified configuration file.
用于训练模型的脚本。
Designed to run from the command line.
设计为从命令行运行。
"""

import os
from os.path import join as pjoin
import sys
import argparse
import matplotlib.pyplot as plt
import zhplot

import torch
from ultralytics import YOLO
import detectron2
import ultralytics.data.build as build

from constants import MODEL_DIR, DATA_DIR, EXPERIMENT_DIR
from yolo_balanced_dataloader import YOLOWeightedDataset


def train_yolo(
        config_fpath: str, 
        model: str, 
        patience: int = 20,
        balanced_dataloader: bool = False,
    ):
    """Train a YOLO model with the specified configuration file and model path.

    Args:
        config_fpath (str): Path to the configuration yaml file.
        model (str): Path to the model pt file.
        patience (int, optional): Number of epochs to run without improvement. Defaults to 20.
        balanced_dataloader (bool, optional): Use a data loader that handles class imbalance.
        Defaults to False.
    """
    save_dir: str = pjoin(
        EXPERIMENT_DIR,
        os.path.basename(config_fpath).split(".")[0] 
            if not balanced_dataloader 
            else "balanced_" + os.path.basename(config_fpath).split(".")[0],
        os.path.basename(model).split(".")[0],
    )
    
    if balanced_dataloader:
        # monkey patch the YOLODataset class
        build.YOLODataset = YOLOWeightedDataset
    
    model = YOLO(model)  # Load a pretrained YOLOv8 model
    model.train(
        data=config_fpath,
        epochs=200,
        imgsz=1024,
        batch=8,
        device=0,
        patience=patience,
        project=save_dir,
    )


def train_yolo_resume(config_fpath: str, model: str, patience: int = 20):
    """
    Train a YOLO model with the specified configuration file and model path.
    Args:
        config_fpath (str): Path to the configuration file.
        model (str): Path to the model file.
        early_stopping (bool): Whether to use early stopping. Default is False.
    """
    model = YOLO(model)
    model.train(
        data=config_fpath,
        epochs=200,
        imgsz=1024,
        batch=12,
        device=0,
        resume=True,
        patience=patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to the model file."
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume training from the last checkmapoint.",
    )
    parser.add_argument(
        "-b",
        "--balanced",
        action="store_true",
        help="Use a data loader that handles class imbalance.",
    )
    args = parser.parse_args()
    

    config_path = pjoin(DATA_DIR, args.config)

    if args.resume:
        # Check if the model file exists
        model_path = pjoin(EXPERIMENT_DIR, args.model, "weights", "last.pt")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist.")
            sys.exit(1)
        print("Resuming training...")
        train_yolo_resume(config_fpath=config_path, model=model_path)
        exit(0)

    model_path = pjoin(MODEL_DIR, args.model)

    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Config file {args.config} does not exist.")
        sys.exit(1)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file {args.model} does not exist.")
        sys.exit(1)
    if "yolo" in model_path:
        print("Training YOLO model...")
        train_yolo(
            config_fpath=config_path, 
            model=model_path, 
            balanced_dataloader=args.balanced
        )
