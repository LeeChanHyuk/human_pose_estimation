import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

video_path = 'C:/Users/user/Desktop/smv.MP4'
save_path = 'C:/Users/user/Desktop/converted_video'
rgb_cap = cv2.VideoCapture(video_path)
w = round(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = rgb_cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
rgb_video_name = video_path.split('/')[-1]
rgb_video_name = rgb_video_name[0:len(rgb_video_name)-4] + '.avi'

median_3= cv2.VideoWriter(
	os.path.join(save_path, 'median_3_'+ rgb_video_name)
	, fourcc, fps, (w, h))

median_5= cv2.VideoWriter(
	os.path.join(save_path, 'median_5_' + rgb_video_name)
	, fourcc, fps, (w, h))
	
bilateral_3 = cv2.VideoWriter(
	os.path.join(save_path, 'bilateral_3_' + rgb_video_name)
	, fourcc, fps, (w, h))

bilateral_5 = cv2.VideoWriter(
	os.path.join(save_path, 'bilateral_5_' + rgb_video_name)
	, fourcc, fps, (w, h))

bilateral_9 = cv2.VideoWriter(
	os.path.join(save_path, 'bilateral_9_' + rgb_video_name)
	, fourcc, fps, (w, h))
while True:
	ret, frame = rgb_cap.read()
	if frame is None:
		break
	median_blur_3 = cv2.medianBlur(frame,3)
	median_3.write(median_blur_3.copy())
	median_blur_5 = cv2.medianBlur(frame,5)
	median_5.write(median_blur_3.copy())

	bilateral_blur3 = cv2.bilateralFilter(frame,3,75,75)
	bilateral_3.write(bilateral_blur3.copy())
	bilateral_blur5 = cv2.bilateralFilter(frame,5,75,75)
	bilateral_5.write(bilateral_blur5.copy())
	bilateral_blur9 = cv2.bilateralFilter(frame,9,75,75)
	bilateral_9.write(bilateral_blur5.copy())

median_3.release()
median_5.release()

bilateral_3.release()
bilateral_5.release()
bilateral_9.release()