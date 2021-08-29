################################################################################
# widescreenify.py
# Aidan Maycock
################################################################################

import cv2
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image

MAX_FRAMES = 100


# Ingest video clip for training
video = cv2.VideoCapture("D:\HDD Downloads\Media\Avatar - The Legend of Korra\Season 1\The.Legend.of.Korra.S01E01-E02.720p.HDTV.x264-HWE.mkv")

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = video.get(cv2.CAP_PROP_FPS)

frame_count = min(frame_count, MAX_FRAMES)

video_array = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frame_count and ret):
    ret, video_array[fc] = video.read()
    fc += 1

video.release()

print(f"{frame_width}x{frame_height}, {frame_count} frames at {frame_rate}fps")


# Split video clip into training input and result
input_clip = np.empty((frame_count, frame_height, ((frame_width // 16) * 12), 3), np.dtype('uint8'))
width_diff = frame_width - ((frame_width // 16) * 12)
width_start = width_diff // 2
width_end = frame_width - (width_diff // 2)

for i in range(0, frame_count):
    input_clip[i] = video_array[i, :, width_start:width_end, :]

target_clip = video_array

print(input_clip.shape)
print(target_clip.shape)

# input_image = Image.fromarray(input_clip[frame_count-1], 'RGB')
# target_image = Image.fromarray(target_clip[frame_count-1], 'RGB')
# input_image.show()
# target_image.show()


# Normalize clips to range of 0 to 1
# input_clip = input_clip / 255.0
# target_clip = target_clip / 255.0