import numpy as np
import utils
import cv2

width = 640
height = 480
yaw=10
pitch=20
roll=30
zero_array = np.zeros((height, width, 3))
utils.draw_axis(zero_array, yaw, pitch, roll, [(width / 20) * 2, (height/10) * 1], color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
cv2.putText(zero_array, '[Body position]:', (int((width/40) * 1), int((height/10) * 4)), 1, 2, (255, 255, 0), 3)
cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 1), int((height/10) * 5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, 'Pitch ::' + str(pitch), (int((width/40) * 1), int((height/10) * 5)), 1, 1.8, (255, 255, 0), 3)
cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 1), int((height/10) * 5)), 1, 1.8, (255, 255, 0), 3)

cv2.imshow("zero_array", zero_array)
cv2.waitKey(0)