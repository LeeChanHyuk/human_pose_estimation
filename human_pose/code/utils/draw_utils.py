from math import *
import cv2
import numpy as np

def draw_axis(img, yaw, pitch, roll, visualization_point, size = 50, color1=(255,0,0), color2=(0,255,0), color3=(0,0,255)):
    pitch = (pitch * np.pi / 180)
    yaw = -(yaw * np.pi / 180)
    roll = (roll * np.pi / 180)

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll))
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw))

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll))
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll))

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw))
    y3 = size * (-cos(yaw) * sin(pitch))

    cv2.line(img, (int(visualization_point[0]), int(visualization_point[1])), (int(visualization_point[0] + x1),int(visualization_point[1] + y1)),color1,3)
    cv2.line(img, (int(visualization_point[0]), int(visualization_point[1])), (int(visualization_point[0] + x2),int(visualization_point[1] + y2)),color2,3)
    cv2.line(img, (int(visualization_point[0]), int(visualization_point[1])), (int(visualization_point[0] + x3),int(visualization_point[1] + y3)),color3,2)

    return img