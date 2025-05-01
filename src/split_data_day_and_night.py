import os
import sys
from shutil import move

import supervision as sv
import cv2

from colour_vs_infrared import is_infrared
from constants import PROJECT_ROOT, DATA_DIR

wfs_data_root = os.path.join(DATA_DIR, "wfs-org")

# load the coco annotations
coco = sv.DetectionDataset.from_coco(
    annotations_path=os.path.join(wfs_data_root, "annotations", "instances.json"),
    images_directory_path=os.path.join(wfs_data_root, "images"),
)


from tqdm import tqdm

day = []
night = []

os.makedirs(os.path.join(wfs_data_root, "day"), exist_ok=True)
os.makedirs(os.path.join(wfs_data_root, "night"), exist_ok=True)

for image in tqdm(coco, desc="Processing images", unit="image"):
    image_path = image[0]
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist.")
        continue
    if is_infrared(image_path):
        night.append(image)
        move(image_path, os.path.join(wfs_data_root, "night", os.path.basename(image_path)))
    else:
        day.append(image)
        move(image_path, os.path.join(wfs_data_root, "day", os.path.basename(image_path)))

# save the day and night annotations into separate files
day_annotations = sv.DetectionDataset.as_coco(
    dataset=day,
    output_path=os.path.join(wfs_data_root, "annotations", "day_instances.json"),
    images_directory_path=os.path.join(wfs_data_root, "day"),
)
night_annotations = sv.DetectionDataset.as_coco(
    dataset=night,
    output_path=os.path.join(wfs_data_root, "annotations", "night_instances.json"),
    images_directory_path=os.path.join(wfs_data_root, "night"),
)