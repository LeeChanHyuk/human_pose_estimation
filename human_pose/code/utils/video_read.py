import cv2
import os

base_path = os.getcwd()
video_path = os.path.join(base_path, 'training_dataset.avi')
cap = cv2.VideoCapture(video_path)
while 1:
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)