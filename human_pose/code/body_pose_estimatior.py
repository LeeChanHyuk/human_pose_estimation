from turtle import left
import cv2
import mediapipe as mp
import math
import numpy as np
from pandas import pivot

from torch import normal

def new_pose_calculator(left_shoulder, right_shoulder, center_stomach):
    center_shoulder = (left_shoulder + right_shoulder) / 2
    yaw, pitch, roll = 0, 0, 0
    # Yaw
    if left_shoulder[2] > right_shoulder[2]: # yaw - direction
        direction_vector = left_shoulder - center_shoulder
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [-1, 0]
        theta = np.inner(direction_vector, pivot_vector) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = np.arccos(theta)
        yaw = -theta
    else:
        direction_vector = right_shoulder - center_shoulder
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [1, 0]
        theta = np.inner(direction_vector, pivot_vector) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = np.arccos(theta)
        yaw = -theta

    # Pitch
    if center_shoulder[2] < center_stomach[2]: # pitch (-) direction
        direction_vector = (center_shoulder - center_stomach)
        direction_vector = (direction_vector[1], direction_vector[2])
        pivot_vector = [1, 0]
        theta = np.inner(direction_vector, pivot_vector) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = np.arccos(theta)
        pitch = theta
    else:
        direction_vector = (center_shoulder - center_stomach)
        direction_vector = (direction_vector[1], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        pitch = (theta * -1)

    # Roll
    if left_shoulder[1] < right_shoulder[1]: # roll+
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[1])
        pivot_vector = [-1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = -theta
    else:
        direction_vector = (right_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[1])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = -theta
    return yaw, pitch, roll


def upside_body_pose_calculator(left_shoulder, right_shoulder, center_stomach):
    center_shoulder = (left_shoulder + right_shoulder) / 2
    yaw, pitch, roll = 0, 0, 0
    # Yaw
    if left_shoulder[2] > right_shoulder[2]: # yaw (-) direction
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [-1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        yaw = -1 * theta
    else:
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        yaw = theta
    # Pitch
    if center_shoulder[2] < center_stomach[2]: # pitch (-) direction
        direction_vector = (center_shoulder - center_stomach)
        direction_vector = (direction_vector[1], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        pitch = (theta * -1)
    else:
        direction_vector = (center_shoulder - center_stomach)
        direction_vector = (direction_vector[1], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        pitch = (theta * -1)
    """# Roll
    if center_shoulder[2] < center_stomach[2]: # pitch (-) direction
        direction_vector = (center_shoulder - center_stomach)
        direction_vector = (direction_vector[0], direction_vector[1])
        pivot_vector = [0, 1]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = (theta * -1)
    else:
        direction_vector = (center_shoulder - center_stomach)
        direction_vector = (direction_vector[0], direction_vector[1])
        pivot_vector = [0, 1]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = theta"""

    if left_shoulder[1] < right_shoulder[1]: # roll+
        direction_vector = (left_shoulder - center_shoulder)
        pivot_vector = [-1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = theta
    else:
        direction_vector = (right_shoulder - center_shoulder)
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = theta
    return yaw, pitch, roll



def body_keypoint_extractor(body_landmarks, landmark_names, depth, width, height):

    """Params
    body_landmarks : body_landmarks from the mediapipe's body keypoint tracking algorithm
    landmark_names : landmark_name from mediapipe
    depth : depth image
    width : image width
    height : image height
    """

    # face landmarks
    nose = body_landmarks[landmark_names.index('nose')]
    left_ear = body_landmarks[landmark_names.index('left_ear')]
    right_ear = body_landmarks[landmark_names.index('right_ear')]
    mouth_left = body_landmarks[landmark_names.index('mouth_left')]
    mouth_right = body_landmarks[landmark_names.index('mouth_right')]

    # Calculate down-side body pose
    left_hip = body_landmarks[landmark_names.index('left_hip')]
    right_hip = body_landmarks[landmark_names.index('right_hip')]

    # Calculate up-side body pose
    left_shoulder = body_landmarks[landmark_names.index('left_shoulder')]
    right_shoulder = body_landmarks[landmark_names.index('right_shoulder')]

    left_eye = body_landmarks[landmark_names.index('left_eye')]
    right_eye = body_landmarks[landmark_names.index('right_eye')]
    center_eye = (left_eye + right_eye) / 2

    # Change z-position from the Depth image because the original z-position is estimated position from face pose 
    # offset is the margin of shoulder position
    left_y_offset = 10
    left_x_offset = 20
    right_x_offset = 20
    right_y_offset = 10
    nose[2] = depth[min(int(nose[1]), height-1), min(width-1, int(nose[0]))]
    left_ear[2] = depth[min(int(left_ear[1])+left_y_offset, height-1), min(width-1, int(left_ear[0])-left_x_offset)]
    right_ear[2] = depth[min(int(right_ear[1])+right_y_offset, height-1), min(width-1, max(0, int(right_ear[0])+right_x_offset))]
    mouth_left[2] = depth[min(int(mouth_left[1])+left_y_offset, height-1), min(width-1, int(mouth_left[0])-left_x_offset)]
    mouth_right[2] = depth[min(int(mouth_right[1])+right_y_offset, height-1), min(width-1, max(0, int(mouth_right[0])+right_x_offset))]
    left_shoulder[2] = depth[min(int(left_shoulder[1])+left_y_offset, height-1), min(width-1, int(left_shoulder[0])-left_x_offset)]
    right_shoulder[2] = depth[min(int(right_shoulder[1])+right_y_offset, height-1), min(width-1, max(0, int(right_shoulder[0])+right_x_offset))]

    left_hip[2] = depth[min(int(left_hip[1])+left_y_offset, height-1), min(width-1, int(left_hip[0])-left_x_offset)]
    right_hip[2] = depth[min(int(right_hip[1])+right_y_offset, height-1), min(width-1, max(0, int(right_hip[0])+right_x_offset))]
    
    center_hip = (left_hip + right_hip) / 2
    center_stomach = [int(max(0, min(center_hip[0], width-1))),int(max(0, min((center_hip[1] * 1 + (left_shoulder[1] + right_shoulder[1]) * 2)/3, height-1))), 0]
    center_stomach[2] = depth[center_stomach[1], center_stomach[0]]
    center_mouth = (mouth_left + mouth_right) / 2

    return left_shoulder, right_shoulder, center_stomach, center_mouth, left_x_offset, left_y_offset, right_x_offset, right_y_offset, center_eye