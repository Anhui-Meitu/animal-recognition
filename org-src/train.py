from argparse import ArgumentParser, BooleanOptionalAction
from os.path import join as pjoin

from ultralytics import YOLO
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

if __name__ == '__main__':
    # cmd line arguments: data, image_size, patience, batch_size, epochs, model
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='wfs', help='数据集, 默认animal', choices=['animal', 'wfs', 'shitai','yanshi','wfs-dataset','fuhe'])
    parser.add_argument('-s', '--imgsz', type=int, default=736, help='image size, 默认736', choices=[320, 416, 512, 640, 736, 896, 1280])
    parser.add_argument('-p', '--patience', type=int, default=30, help='early stopping patience, 默认20')
    parser.add_argument('-b', '--batch', type=int, default=16, help='batch size, 默认32')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='epochs, 默认200')
    parser.add_argument('-m', '--model', type=str, default='l', help='model, 默认l', choices=['s', 'm', 'l', 'x'])
    parser.add_argument('--pretrained', action=BooleanOptionalAction, help='是否使用预训练模型, 默认True')

    data = pjoin(parser.parse_args().dataset, 'data.yaml')

    project = pjoin('runs', 'train', 'cleaned-data')
    experiment = f'model_{parser.parse_args().model}_size_{parser.parse_args().imgsz}_{parser.parse_args().dataset}_pretrain-{parser.parse_args().pretrained}_with_aug'
    batch = parser.parse_args().batch

    # 训练模型
    if parser.parse_args().pretrained:
        model = YOLO(f'yolov8{parser.parse_args().model}.pt')
    else:  
        model = YOLO(f'yolov8{parser.parse_args().model}.yaml')
    # save training results to performance/ directory
    result = model.train(data=pjoin('dataset', data), imgsz=parser.parse_args().imgsz, patience=parser.parse_args().patience, batch=batch, epochs=parser.parse_args().epochs, project=project, name=experiment)