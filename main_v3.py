import cv2
import torch
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import keras
from moviepy.editor import *
import object_detection
# #adding yolo model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # change to yolov5x
model = object_detection()

video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret,frame = cap.read()
    results = model(frame)
    mat_results = np.squeeze(results.render())

    cv2.imshow("yolo", mat_results)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()