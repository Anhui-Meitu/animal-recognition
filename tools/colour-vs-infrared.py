"""split files in directory into colour and infrared images
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
            
if __name__ == "__main__":
    # supply directory path to images
    if len(sys.argv) != 2:
        print("Usage: python split_images_into_folders.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    os.chdir(directory_path)
    image_files = glob('*.jpg')
    image_files_count = len(image_files)
    print(f"Found {image_files_count} images")
    if image_files_count == 0:
        print("No images found in the directory.")
        sys.exit(1)
    split_images_into_folders(image_files)
    print("Image splitting completed.")
