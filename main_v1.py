import cv2
import torch
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import os
import keras
from moviepy.editor import *
#adding yolo model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # change to yolov5x


video_path = "video.mp4"
video_clip = VideoFileClip(video_path)

def detect_and_render(frame):
    results = model(frame)
    rendered_frame = np.squeeze(results.render())
    return rendered_frame




# Process each frame and create a new video clip with rendered objects
rendered_frames = []
for frame in video_clip.iter_frames(fps=video_clip.fps, dtype='uint8'):
    rendered_frame = detect_and_render(frame)
    rendered_frames.append(rendered_frame)

# Create a new video clip with the rendered frames
rendered_video_clip = ImageSequenceClip(rendered_frames, fps=video_clip.fps)

# Save the new video clip with rendered objects
output_path = 'output_video_with_objects.mp4'
rendered_video_clip.write_videofile(output_path, codec='libx264')

# Close the video clips
video_clip.reader.close()
rendered_video_clip.reader.close()

print("Object detection and rendering completed!")






