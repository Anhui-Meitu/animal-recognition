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
    parser.add_argument('-t', '--date', type=str, required=False, help='Date in YYYY-MM-DD format')
    args = parser.parse_args()
    
    if not args.date:
        # If no date is provided, try to find the last date the script was run
        if not os.path.exists('last_run_date.json'):
            print("No date provided and last_run_date.json not found. Exiting.")
            exit(1)
        with open('last_run_date.json', 'r') as f:
            last_run_date = f.read()
        last_run_date = re.search(r'\d{4}-\d{2}-\d{2}', last_run_date).group(0)
        # use the day before the last run date
        last_run_date = (
            datetime.datetime.strptime(last_run_date, '%Y-%m-%d') - 
            datetime.timedelta(days=1)
        ).strftime('%Y-%m-%d')
        print(f"No date provided. Using last run date: {last_run_date}")
        args.date = last_run_date

    # Convert the date string to a datetime object
    date = datetime.datetime.strptime(args.date, '%Y-%m-%d')
    
    # save to output dir
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/videos', exist_ok=True)
    os.makedirs('output/images', exist_ok=True)
    
    for dir_name, videos, images in tqdm(find_files_after_date(args.directory, date), desc="Finding files"):
        print(dir_name)
        # Copy video files
        for video in videos:
            dest = os.path.join('output/videos', os.path.basename(video))
            copyfile(video, dest)
        
        # Copy image files
        for image in images:
            dest = os.path.join('output/images', os.path.basename(image))
            copyfile(image, dest)
    print("Files copied to output directory.")
    
    # save current date to JSON file
    with open('last_run_date.json', 'w') as f:
        f.write(f'{{"last_run_date": "{datetime.datetime.now().strftime("%Y-%m-%d")}"}}')
    
    