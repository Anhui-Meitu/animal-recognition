"""separate colour from infrared images
"""
from glob import glob
import cv2
import os
import sys
from shutil import move

def is_infrared(image_path):
    """
    check if image is black and white shot by infrared camera
    All images are RGB though they may be black and white
    """
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is grayscale (single channel)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return True

    # Check if the image is black and white (R, G, B channels are equal)
    b, g, r = cv2.split(image)
    if (b == g).all() and (g == r).all():
        return True

    return False
    

def split_images_into_folders(image_files):
    """
    split images into colour and infrared folders
    """
    # Create directories for colour and infrared images
    os.makedirs('colour', exist_ok=True)
    os.makedirs('infrared', exist_ok=True)

    # Loop through the image files and move them to the appropriate folder
    for image_file in image_files:
        try:
            if is_infrared(image_file):
                move(image_file, os.path.join('infrared', image_file))
                print(f"Moved {image_file} to infrared")
            else:
                move(image_file, os.path.join('colour', image_file))
                print(f"Moved {image_file} to colour")
        except Exception as e:
            print(f"Error moving file {image_file}: {e}")
            

