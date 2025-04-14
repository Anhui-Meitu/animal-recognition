# split-into-folders.py
from glob import glob
import os

# split directory into five folders
image_files = glob('*.jpg')
image_files_count = len(image_files)
print(f"Found {image_files_count} images")

image_batches = [image_files[i:i + image_files_count // 5] for i in range(0, image_files_count, image_files_count // 5)]
print(f"Splitting into {len(image_batches)} batches")

for i, batch in enumerate(image_batches):
    folder_name = f"batch_{i}"
    os.makedirs(folder_name, exist_ok=True)
    for image_file in batch:
        os.rename(image_file, os.path.join(folder_name, image_file))
        print(f"Moved {image_file} to {folder_name}")
    