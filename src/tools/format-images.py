"""convert image formats to uniform jpg format
"""

import os
import sys
from glob import glob
from PIL import Image
from tqdm import tqdm
import cv2
import argparse

def convert_images_to_jpg(input_dir, output_dir):
    """
    Convert all images in the input directory to JPG format and save them in the output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the input directory
    image_files = glob(os.path.join(input_dir, '*'))

    # Loop through each image file
    for image_file in tqdm(image_files):
        if image_file.lower().endswith('jpg'):
            # Skip jpg files
            continue
        if image_file.lower().endswith('txt'):
            # Skip text files
            continue
        try:
            # Open the image using PIL
            with Image.open(image_file) as img:
                # Convert to RGB (in case it's RGBA or other formats)
                img = img.convert('RGB')
                # Save the image as JPG in the output directory
                output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '.jpg')
                img.save(output_file, 'JPEG')
        except Exception as e:
            print(f"Error converting {image_file}: {e}")
            continue
        # Optionally, you can also use OpenCV to read and save the image
        # img_cv = cv2.imread(image_file)
        # output_file_cv = os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '.jpg')
        # cv2.imwrite(output_file_cv, img_cv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert images into uniform jpg formats"
    )
    
    parser.add_argument(
        "-d", "--input_dir", type=str, required=True,
        help="Path to the input directory"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=False,
        help="Path to the output directory, optional, use input directory if not provided"
    )
    args = parser.parse_args()
    
    input_directory = args.d
    output_directory = args.o if args.o else input_directory

    # Convert images to JPG format
    convert_images_to_jpg(input_directory, output_directory)