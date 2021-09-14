################################################################################
# widescreenify.py
# Aidan Maycock
# Heavily based on a Pix2Pix implementation by Arthur Juliani
# https://github.com/awjuliani/Pix2Pix-Film/blob/master/Pix2Pix.ipynb
################################################################################

import argparse
import os

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
import cv2

# PARAMS
FRAME_GROUP = 100
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280

H_RATIO = 9
W_RATIO = 16
I_RATIO = 12

# GLOBALS

################################################################################
# USER FUNCTIONS
################################################################################

# Check video properties to ensure it can be trained/tested with the model.
def check_props(width, height, mode):
    if (mode == "train"):
        if not ((FRAME_WIDTH == width) and (FRAME_HEIGHT == height)):
            return False
    else:
        if not ((((FRAME_WIDTH // W_RATIO) * I_RATIO) == width) and (FRAME_HEIGHT == height)):
            return False

    return True


# Ingest video and create train and test frame group
# output_frames will return None if mode is test
def get_frame_group(video, current_frame, stop_frame, mode):
    group_stop_frame = current_frame + FRAME_GROUP if ((current_frame + FRAME_GROUP) <= stop_frame) else stop_frame
    group_frame_count = group_stop_frame - current_frame

    width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
    width_start = width_diff // 2
    width_end = FRAME_WIDTH - (width_diff // 2)

    input_frames = np.empty((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), np.dtype('uint8'))
    if (mode == "train"):
        output_frames = np.empty((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), np.dtype('uint8'))

    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    for frame_index in range(0, group_frame_count):
        ret, frame = video.read()

        if not ret:
            print("Video read error")
            exit(0)

        input_frames[frame_index, :, width_start:width_end, :] = frame[:, width_start:width_end, :]
        if (mode == "train"):
            output_frames[frame_index] = frame

    if (mode == "train"):
        return group_stop_frame, input_frames, output_frames
    else:
        return group_stop_frame, input_frames, None



################################################################################
# SCRIPT START
################################################################################

# Ingest command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", type=str, required=True, help="Set path to video file for training or testing.")
parser.add_argument("-m", "--mode", type=str, default="train", help="Set to either train or test mode.")
parser.add_argument("-c", "--ckpt", type=str, default="test.ckpt", help="Set path to model checkpoint file.")
parser.add_argument("-n", "--new_ckpt", type=str, help="Set path to new model checkpoint file, to avoid overwriting previous model. If not included imported checkpoint will be overwritten.")
parser.add_argument("-s", "--start_frame", type=int, default=0, help="Set the starting frame for video ingest.")
parser.add_argument("-f", "--stop_frame", type=int, help="Set maximum number of frames to train or test by indicating the stop frame.")

args = parser.parse_args()

video_path = args.path
ckpt_path = args.ckpt
new_ckpt_path = args.new_ckpt
mode = args.mode
stop_frame = args.stop_frame
current_frame = args.start_frame

# Ingest video clip for training
# Initial Video clip: "D:\HDD Downloads\Media\Avatar - The Legend of Korra\Season 1\The.Legend.of.Korra.S01E01-E02.720p.HDTV.x264-HWE.mkv"
video = cv2.VideoCapture(video_path)

video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # in frames
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = video.get(cv2.CAP_PROP_FPS)

print(f"{frame_width}x{frame_height}, {video_length} frames at {frame_rate}fps")

if (stop_frame != None):
    stop_frame = min(video_length, stop_frame)
else:
    stop_frame = video_length

if (current_frame >= stop_frame):
    print(f"Start frame is after stop frame:: Start:{current_frame} Stop:{stop_frame}")
    exit(1)

if not check_props(frame_width, frame_height, mode):
    print(f"Incompatible input video dimensions:: {frame_width}x{frame_height}")
    exit(1)

current_frame, input_frames, output_frames = get_frame_group(video, current_frame, stop_frame, mode)

print(input_frames.shape)
print(output_frames.shape)
print(current_frame)

current_frame, input_frames, output_frames = get_frame_group(video, current_frame, stop_frame, mode)

print(input_frames.shape)
print(output_frames.shape)
print(current_frame)

video.release()

print(input_frames.shape)
print(output_frames.shape)

# input_image = Image.fromarray(input_clip[FRAME_GROUP-1], 'RGB')
# target_image = Image.fromarray(target_clip[FRAME_GROUP-1], 'RGB')
# input_image.show()
# target_image.show()


# Normalize clips to range of 0 to 1
# input_clip = input_clip / 255.0
# target_clip = target_clip / 255.0

# Define model and layers
