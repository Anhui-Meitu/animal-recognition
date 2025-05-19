# chop videos into images
from glob import glob
import cv2
import os
import argparse

def chop_videos_into_images(video_files_dir, output_dir, interval=1):
    """
    chop videos into images
    """
    # Create directories for images
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the video files and extract frames
    for video_file in glob(os.path.join(video_files_dir, '*.mp4')):
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_file)
            frame_count = 0
            
            # Loop through the frames of the video
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                
                # If the frame was not read successfully, break the loop
                if not ret:
                    break
                                
                # Save the frame as an image every 'interval' seconds
                if frame_count % (int(cap.get(cv2.CAP_PROP_FPS)) * interval) == 0:
                    image_file = os.path.join(output_dir, f"{os.path.basename(video_file).split('.')[0]}-frame-{frame_count}.jpg")
                    cv2.imwrite(image_file, frame)
                    print(f"Saved {image_file}")
                
                frame_count += 1
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
        finally:
            # Release the video capture object
            cap.release()
            cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Chop videos into images")
    arg_parser.add_argument("-d", "--video_files_dir", type=str, required=True, help="Directory containing video files")
    # arg_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images")
    arg_parser.add_argument("-i", "--interval", type=int, default=1, help="Interval in seconds to save images")
    args = arg_parser.parse_args()
    
    if not os.path.exists(args.video_files_dir):
        print(f"Video files directory {args.video_files_dir} does not exist.")
        exit(1)
    # if not os.path.exists(args.output_dir):
    #     print(f"Output directory {args.output_dir} does not exist.")
    #     exit(1)
    output_dir = os.path.join(args.video_files_dir, 'chopped_images')
    
    chop_videos_into_images(args.video_files_dir, output_dir, args.interval)