import numpy as np
import utils
import cv2

width = 1280
height = 480
yaw=10
pitch=20
roll=30
zero_array = np.zeros((height, width, 3))
utils.draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 4, (height/20) * 1.5], color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
cv2.putText(zero_array, '[Body position]:', (int((width/40) * 1), int((height/20) * 6)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' X :' + str(yaw), (int((width/40) * 1), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Y :' + str(pitch), (int((width/40) * 1), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Z :' + str(roll), (int((width/40) * 1), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, '[Body pose]:', (int((width/40) * 1), int((height/20) * 13)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 1), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 1), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 1), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)


utils.draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 18.5, (height/20) * 1.5], color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
cv2.putText(zero_array, '[Face position]:', (int((width/40) * 15), int((height/20) * 6)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' X :' + str(yaw), (int((width/40) * 15), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Y :' + str(pitch), (int((width/40) * 15), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Z :' + str(roll), (int((width/40) * 15), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, '[Face pose]:', (int((width/40) * 15), int((height/20) * 13)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 15), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 15), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 15), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)


utils.draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 33, (height/20) * 1.5], color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
cv2.putText(zero_array, '[Body position]:', (int((width/40) * 29), int((height/20) * 6)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' X :' + str(yaw), (int((width/40) * 29), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Y :' + str(pitch), (int((width/40) * 29), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Z :' + str(roll), (int((width/40) * 29), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, '[Body pose]:', (int((width/40) * 29), int((height/20) * 13)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 29), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 29), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 29), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)

cv2.imshow("zero_array", zero_array)
cv2.waitKey(0)