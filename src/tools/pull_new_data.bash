# # calling the tools to process new data from aiqx platform
# # make sure we are in the base conda environment
# conda activate base

# # pull data from the aiqx platform
# python find_file_after_date.py -d /media/meitu001/Data/aiqx/img/

# find the folder contains today's date in YYYYMMDD format
today_folder=$(ls -d /media/meitu001/Data/share/*-$(date +%Y%m%d) | tail -n 1)

# # chop the videos into images
# python chop_video.py -d $today_folder/videos --interval 2 # 2 seconds interval in the video

# # move the images to the images folder
# mv $today_folder/videos/chopped_images/* $today_folder/images/

# # day night separation
# python colour_vs_infrared.py -d $today_folder/images/

# add the day and night images folder to zip
zip -r $today_folder-colour.zip $today_folder-colour
zip -r $today_folder-infrared.zip $today_folder-infrared
# # remove the day and night images folder
# # rm -rf $today_folder/($today_folder)-day
# # rm -rf $today_folder/($today_folder)-night
# zip today's folder
zip -r $today_folder.zip $today_folder
# # remove today's folder
# rm -rf $today_folder