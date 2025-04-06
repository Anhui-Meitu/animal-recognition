"""convert image formats to uniform jpg format
"""

import os
import sys
from glob import glob
from PIL import Image
from tqdm import tqdm
import cv2

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
    # Check if the input directory is passed as an argument
    if len(sys.argv) < 3:
        print("Usage: python convert_images.py <input_directory> <output_directory>")
        sys.exit(1)

    # Get the input and output directories from the command line arguments
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Convert images to JPG format
    convert_images_to_jpg(input_directory, output_directory)