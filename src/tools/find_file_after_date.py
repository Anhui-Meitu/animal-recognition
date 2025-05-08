# Find tasks created after a particular date to
# retrieve new data for labelling
import os
import datetime
import argparse
from shutil import copyfile
from tqdm import tqdm
from glob import glob
import re

def find_files_after_date(directory, date):
    # directories are named as taskYYYYMMDD*
    date_str = date.strftime('%Y%m%d')
    
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            # Check if the directory name matches the pattern
            match = re.match(r'task(\d{8})', dir_name)
            if match:
                dir_date_str = match.group(1)
                dir_date = datetime.datetime.strptime(dir_date_str, '%Y%m%d')
                
                # Compare the dates
                if dir_date > date:
                    # Check for video and image files in the directory
                    videos = glob(os.path.join(root, dir_name, '*.mp4'))
                    images = glob(os.path.join(root, dir_name, '*.jpg'))
                    if videos or images:
                        yield dir_name, videos, images
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find files created after a specific date.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory to search in')
    parser.add_argument('-t', '--date', type=str, required=True, help='Date in YYYY-MM-DD format')
    args = parser.parse_args()

    # Convert the date string to a datetime object
    date = datetime.datetime.strptime(args.date, '%Y-%m-%d')
    
    # save to output dir
    os.mkdir('output')
    os.mkdir('output/videos')
    os.mkdir('output/images')
    
    for dir_name, videos, images in tqdm(find_files_after_date(args.directory, date), desc="Finding files"):
        # Copy video files
        for video in videos:
            dest = os.path.join('output/videos', os.path.basename(video))
            copyfile(video, dest)
        
        # Copy image files
        for image in images:
            dest = os.path.join('output/images', os.path.basename(image))
            copyfile(image, dest)
    print("Files copied to output directory.")
    
    