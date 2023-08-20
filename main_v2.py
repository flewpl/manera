import cv2
import torch
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import os
import keras

model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # change to yolov5x

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)


    cv2.imshow("TOLO",np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


