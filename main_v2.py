
import cv2
import torch
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import os
import keras

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # change to yolov5x

video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)
output_path = "output_video_with_objects.mp4"

# Get original video's FPS and frame dimensions
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and VideoWriter for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))



while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    
    rendered_frame = np.squeeze(results.render())
    out.write(rendered_frame)

cap.release()
out.release()
cv2.destroyAllWindows()


# 23 sec to release