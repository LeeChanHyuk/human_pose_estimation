import numpy as np
import utils.draw_utils as draw_utils
import cv2
from gaze_estimation_module.util.gaze import draw_gaze

def draw_body_information(zero_array, width, height, x, y, z, yaw, pitch, roll): # [width, height, x, y, z, yaw, pitch, roll]
	draw_utils.draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 4, (height/20) * 1.5], color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
	cv2.putText(zero_array, '[Body position]:', (int((width/40) * 1), int((height/20) * 6)), 1, 2, (255, 0, 0), 3)
	cv2.putText(zero_array, ' X :' + str(x), (int((width/40) * 1), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Y :' + str(y), (int((width/40) * 1), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Z :' + str(z), (int((width/40) * 1), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, '[Body pose]:', (int((width/40) * 1), int((height/20) * 13)), 1, 2, (255, 0, 0), 3)
	cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 1), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 1), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 1), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)
	return zero_array


def draw_face_information(zero_array, width, height, x, y, z, yaw, pitch, roll): # [width, height, x, y, z, yaw, pitch, roll]
	draw_utils.draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 18.5, (height/20) * 1.5], color1=(255,0,0), color2=(0,255,0), color3=(0,0,255))
	cv2.putText(zero_array, '[Head position]:', (int((width/40) * 15), int((height/20) * 6)), 1, 2, (0, 255, 0), 3)
	cv2.putText(zero_array, ' X :' + str(x), (int((width/40) * 15), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Y :' + str(y), (int((width/40) * 15), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Z :' + str(z), (int((width/40) * 15), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, '[Head pose]:', (int((width/40) * 15), int((height/20) * 13)), 1, 2, (0, 255, 0), 3)
	cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 15), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 15), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 15), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)
	return zero_array

def draw_gaze_information(zero_array, width, height, x, y, z, gazes): # [width, height, x, y, z, x, y]
	left_gaze = gazes[1]
	right_gaze = gazes[0]
	draw_gaze(zero_array, ((width/40) * 32, (height/20) * 1.5), left_gaze, length=60.0, thickness=2)
	draw_gaze(zero_array, ((width/40) * 34, (height/20) * 1.5), right_gaze, length=60.0, thickness=2)
	cv2.putText(zero_array, '[Eye position]:', (int((width/40) * 29), int((height/20) * 6)), 1, 2, (0, 0, 255), 3)
	cv2.putText(zero_array, ' X :' + str(x), (int((width/40) * 29), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Y :' + str(y), (int((width/40) * 29), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Z :' + str(z), (int((width/40) * 29), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, '[Eye pose]:', (int((width/40) * 29), int((height/20) * 13)), 1, 2, (0, 0, 255), 3)
	cv2.putText(zero_array, ' Left_x:' + str(round(left_gaze[1], 2)), (int((width/40) * 29), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Left_y :' + str(round(left_gaze[0], 2)), (int((width/40) * 29), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Right_x:' + str(round(right_gaze[1], 2)), (int((width/40) * 29), int((height/20) * 19)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Right_y:' + str(round(right_gaze[0], 2)), (int((width/40) * 29), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)
	return zero_array
