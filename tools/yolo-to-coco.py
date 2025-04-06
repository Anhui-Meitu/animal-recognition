"""convert YOLO annotations to COCO format
Usage:
    python tools/yolo-to-coco.py --yolo-dir <path/to/yolo/annotations> --coco-dir <path/to/coco/annotations>
Args:
    yolo_dir (str): path to YOLO annotations directory
    coco_dir (str): path to COCO annotations directory
"""
import os
import sys
import json
from glob import glob

import cv2
import numpy as np
import tqdm
import imagesize

def convert_yolo_to_coco(yolo_dir, coco_dir):
    # Create the COCO directory if it doesn't exist
    os.makedirs(coco_dir, exist_ok=True)

    # Initialize COCO format data structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Get all YOLO annotation files
    yolo_files = glob(os.path.join(yolo_dir, "*-*.txt"))
    class_file = os.path.join(yolo_dir, "classes.txt")
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            classes = f.read().strip().splitlines()
    else:
        classes = []
        print(f"Warning: {class_file} not found. Using default class names.")

    # Initialize category mapping
    category_mapping = {}
    for i, class_name in enumerate(classes):
        category_mapping[i] = i + 1
        # coco category id starts from 1
        coco_data["categories"].append({
            "id": i + 1,
            "name": class_name,
            "supercategory": "none"
        })

    # Process each YOLO annotation file
    for yolo_file in tqdm.tqdm(yolo_files, desc="Processing YOLO files", unit="file"):
        # Get the corresponding image file
        image_file = yolo_file.replace(".txt", ".jpg")
        if not os.path.exists(image_file):
            print(f"Warning: {image_file} not found. Skipping {yolo_file}.")
            continue

        # Get image dimensions
        width, height = imagesize.get(image_file)
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        image_id = image_id.split("-")[-1]  # Extract the image ID from the filename
        # convert the hexadecimal string to an integer
        image_id = int(image_id, 16)

        # Add image info to COCO data
        coco_data["images"].append({
            "id": int(image_id),
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_file)
        })

        # Read YOLO annotations
        with open(yolo_file, 'r') as f:
            yolo_annotations = f.readlines()

        # Process each annotation
        for annotation in yolo_annotations:
            class_id, x_center, y_center, w, h = map(float, annotation.strip().split())
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            # Convert YOLO format to COCO format
            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)

            # Add annotation info to COCO data
            coco_data["annotations"].append({
                "id": len(coco_data["annotations"]) + 1,
                "image_id": int(image_id),
                "category_id": category_mapping[int(class_id)],
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
                "attributes": {}
            })
    # Save COCO annotations to file
    coco_file = os.path.join(coco_dir, "annotations.json")
    with open(coco_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO annotations saved to {coco_file}")


if __name__ == "__main__":
    # Check if the input directory is passed as an argument
    if len(sys.argv) < 3:
        print("Usage: python yolo-to-coco.py --yolo-dir <path/to/yolo/annotations> --coco-dir <path/to/coco/annotations>")
        sys.exit(1)

    # Get the input and output directories from the command line arguments
    yolo_directory = sys.argv[2]
    coco_directory = sys.argv[4]

    # Convert YOLO annotations to COCO format
    convert_yolo_to_coco(yolo_directory, coco_directory)
    print(f"Converted YOLO annotations from {yolo_directory} to COCO format in {coco_directory}")