from argparse import ArgumentParser
from os.path import join as pjoin

from ultralytics import YOLO
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

if __name__ == '__main__':
    # 训练模型
    # transform = YOLO.transforms
    # cmd arguments: data, image_size, patience, batch_size, epochs, model
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='clean', help='dataset.yaml path, clean or hard, 默认clean', choices=['clean', 'hard'])
    parser.add_argument('-s', '--imgsz', type=int, default=640, help='image size, 默认640', choices=[320, 416, 512, 640, 736, 896])
    parser.add_argument('-p', '--patience', type=int, default=20, help='early stopping patience, 默认20')
    parser.add_argument('-b', '--batch', type=int, default=16, help='batch size, 默认32')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='epochs, 默认200')
    parser.add_argument('-m', '--model', type=str, default='m', help='model, 默认m', choices=['s', 'm', 'l', 'x'])

    if parser.parse_args().data == 'clean':
        data = 'data-clean.yaml'
    else:
        data = 'data-hard.yaml'

    project = pjoin('runs', 'detect', 'clean_test')
    experiment = f'model_{parser.parse_args().model}_size_{parser.parse_args().imgsz}_with_aug' if parser.parse_args().data == 'clean' else f'model_{parser.parse_args().model}_size_{parser.parse_args().imgsz}_with_aug_hard'

    if parser.parse_args().imgsz >= 736:
        batch = 16
    else:
        batch = parser.parse_args().batch

    # 训练模型
    model = YOLO(f'yolov8{parser.parse_args().model}.pt')
    # save training results to performance/ directory
    model.tune(data=pjoin('dataset', 'animal', data), imgsz=parser.parse_args().imgsz, batch=batch, epochs=30, iterations=100, optimizer="AdamW", plots=False, save=False, val=True)