"""separate colour from infrared images
"""
from glob import glob
import cv2
import os
import sys
from shutil import move
import argparse

def is_infrared(image_path):
    """
    check if image is black and white shot by infrared camera
    All images are RGB though they may be black and white
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # crop out the centre patch of the image
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    crop_size = min(h, w) // 2
    crop_x1 = center_x - crop_size // 2
    crop_x2 = center_x + crop_size // 2
    crop_y1 = center_y - crop_size // 2
    crop_y2 = center_y + crop_size // 2
    image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Check if the image is grayscale (single channel)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return True

    # Check if the image is black and white (R, G, B channels are equal)
    b, g, r = cv2.split(image)
    if (b == g).all() and (g == r).all():
        return True
    
    # close the image to prevent memory leak
    cv2.destroyAllWindows()
    image = None

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
        finally:
            # close the image to prevent memory leak
            cv2.destroyAllWindows()
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Separate colour and infrared images")
    arg_parser.add_argument("--image_files_dir", type=str, required=True, help="Directory containing image files")
    args = arg_parser.parse_args()
    
    if not os.path.exists(args.image_files_dir):
        print(f"Image files directory {args.image_files_dir} does not exist.")
        exit(1)
        
    os.chdir(args.image_files_dir)
    
    image_files = glob(os.path.join(args.image_files_dir, '*.jpg')) + glob(os.path.join(args.image_files_dir, '*.png'))
    if not image_files:
        print(f"No image files found in {args.image_files_dir}.")
        exit(1)
    if len(image_files) == 0:
        print(f"No image files found in {args.image_files_dir}.")
        exit(1)
    
    split_images_into_folders(image_files)
    print("Finished separating images into colour and infrared folders.")