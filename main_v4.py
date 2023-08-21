import pixellib
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
np.bool = np.bool_
segmentation_model = instance_segmentation()
segmentation_model.load_model('model/rcnn.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Apply instance segmentation
    res = segmentation_model.segmentFrame(frame, show_bboxes=True)
    image = res[1]
    
    cv2.imshow('Instance Segmentation', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()