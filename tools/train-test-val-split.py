#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split images and labels in a given directory into
train test val subdirectories in the ratio of 80:10:10.

Usage:

python move-all.py <directory>
"""

import os
import sys
from glob import glob
import random
from shutil import move
import tqdm

# split files into train test val partition in the ratio of 80:10:10
def split_files(images, labels, train_ratio=0.8, val_ratio=0.1):
    # combine images and labels into pairs
    paired_files = list(zip(images, labels))

    # randomly shuffle the pairs with the same seed
    random.seed(42)
    random.shuffle(paired_files)

    # unzip the shuffled pairs
    images, labels = zip(*paired_files)

    # calculate the number of files in each partition
    total_files = len(images)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    return (
        images[:train_count],
        images[train_count : train_count + val_count],
        images[train_count + val_count :],
        labels[:train_count],
        labels[train_count : train_count + val_count],
        labels[train_count + val_count :],
    )


# move files to the destination directory
def move_files(files, dest_dir):
    # create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # move the files to the destination directory
    for file in tqdm.tqdm(files, desc=f"Moving files to {dest_dir}", unit="file"):
        try:
            move(file, dest_dir)
            print(f"Moved {file} to {dest_dir}")
        except Exception as e:
            print(f"Error moving {file} to {dest_dir}: {e}")


# main function to move files from subdirectories to the parent directory
if __name__ == "__main__":
    # check if the directory is passed as an argument
    if len(sys.argv) < 2:
        print("Usage: python move-all.py <directory>")
        sys.exit(1)

    # get the directory from the command line arguments
    directory = sys.argv[1]

    # get all image files in the directory
    images = (
        glob(os.path.join(directory, "*.jpg"))
        + glob(os.path.join(directory, "*.png"))
        + glob(os.path.join(directory, "*.jpeg"))
    )
    # find their corresponding labels
    labels = images.copy()
    labels = [os.path.splitext(label)[0] + ".txt" for label in labels]

    # split the files
    train_files, val_files, test_files, train_labels, val_labels, test_labels = (
        split_files(images, labels)
    )

    for pair in zip(train_files, train_labels):
        print(f"Image: {pair[0]}, Label: {pair[1]}")
    for pair in zip(val_files, val_labels):
        print(f"Image: {pair[0]}, Label: {pair[1]}")
    for pair in zip(test_files, test_labels):
        print(f"Image: {pair[0]}, Label: {pair[1]}")

    # move the files to the destination directory
    move_files(train_labels, os.path.join(directory, "train"))
    move_files(train_files, os.path.join(directory, "train"))
    move_files(val_labels, os.path.join(directory, "val"))
    move_files(val_files, os.path.join(directory, "val"))
    move_files(test_labels, os.path.join(directory, "test"))
    move_files(test_files, os.path.join(directory, "test"))
