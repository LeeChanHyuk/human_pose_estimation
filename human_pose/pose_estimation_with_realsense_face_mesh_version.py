#!/usr/bin/python3
# -*- coding:utf-8 -*-
##############################################################
################## 2022.01.09.ChanHyukLee ####################
##############################################################
# Facial & Body landmark is from mediaPipe
# Gaze estimation module is from david-wb (https://github.com/david-wb/gaze-estimation)
# Head pose estimation module is from 1996scarlet (https://github.com/1996scarlet/Dense-Head-Pose-Estimation)

import argparse
import math
from ssl import ALERT_DESCRIPTION_NO_RENEGOTIATION
from turtle import right
import head_pose_estimation_module.service as service
from gaze_estimation_module.gaze_estimation import estimate_gaze_from_face_image
import cv2
import time
import os
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

only_detection_mode = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
visualization = True
body_pose_estimation = True 
head_pose_estimation = True # 12 프레임 저하
gaze_estimation = False # 22프레임 저하



landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]


def upside_body_pose_calculator(left_shoulder, right_shoulder, center_hip):
    center_shoulder = (left_shoulder + right_shoulder) / 2
    yaw, pitch, roll = 0, 0, 0
    # Yaw
    if left_shoulder[2] > right_shoulder[2]: # yaw (-) direction
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [-1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        yaw = theta
    else:
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        yaw = (theta * -1)
    # Pitch
    if center_shoulder[2] < center_hip[2]: # pitch (-) direction
        direction_vector = (center_shoulder - center_hip)
        direction_vector = (direction_vector[1], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        pitch = (theta * -1)
    else:
        direction_vector = (center_shoulder - center_hip)
        direction_vector = (direction_vector[1], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        pitch = (theta * -1)
    # Roll
    if center_shoulder[2] < center_hip[2]: # pitch (-) direction
        direction_vector = (center_shoulder - center_hip)
        direction_vector = (direction_vector[0], direction_vector[1])
        pivot_vector = [0, 1]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = (theta * -1)
    else:
        direction_vector = (center_shoulder - center_hip)
        direction_vector = (direction_vector[0], direction_vector[1])
        pivot_vector = [0, 1]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        roll = theta
    return yaw, pitch, roll

def main(color=(224, 255, 255)):
    base_path = os.getcwd()
    
    # Initialization step
    fa = service.DepthFacialLandmarks(os.path.join(base_path, "head_pose_estimation_module/weights/sparse_face.tflite"))
    #fa = service.DenseFaceReconstruction(os.path.join(base_path, "head_pose_estimation_module/weights/dense_face.tflite"))
    print('Head detection module is initialized')

    # Initialinze Head pose estimation object handler
    handler = getattr(service, 'pose')
    print('Head pose module is initialized')

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Define pose estimation & face detection thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0) as pose:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
            #try:
            print('Camera settings is started')
            # Create a context object. This object owns the handles to all connected realsense devices
            pipeline = rs.pipeline()

            # Configure streams
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            pipeline.start(config)

            print('Camera settings is initialized')

            # Load the frame from webcam
            while True:
                start_time = time.time()
                frames = pipeline.wait_for_frames()
                align_frames = align.process(frames)
                frame = align_frames.get_color_frame()
                depth = align_frames.get_depth_frame()
                if not depth or not frame:
                    print('Preparing camera')
                    continue

                # Convert images to numpy arrays
                depth = np.array(depth.get_data())
                frame = np.array(frame.get_data())
                if depth.shape != frame.shape:
                    frame = cv2.resize(frame, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

                # frame shape for return normalized bounding box info
                height, width = frame.shape[:2]

                # Media pipe face detection
                bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(bgr_image)

                # Multi-face detection
                if results.multi_face_landmarks:
                    face_boxes = []
                    left_eye_boxes = []
                    right_eye_boxes = []
                    for face_landmarks in results.multi_face_landmarks:
                        face_x1 = face_landmarks.landmark[234].x * width
                        face_y1 = face_landmarks.landmark[10].y * height
                        face_x2 = face_landmarks.landmark[454].x * width
                        face_y2 = face_landmarks.landmark[152].y * height

                        left_eye_inner_x1 = face_landmarks.landmark[161].x * width
                        left_eye_inner_y1 = face_landmarks.landmark[161].y * height
                        left_eye_inner_x2 = face_landmarks.landmark[154].x * width
                        left_eye_inner_y2 = face_landmarks.landmark[154].y * height
                        right_eye_inner_x1 = face_landmarks.landmark[398].x * width
                        right_eye_inner_y1 = face_landmarks.landmark[398].y * height
                        right_eye_inner_x2 = face_landmarks.landmark[390].x * width
                        right_eye_inner_y2 = face_landmarks.landmark[390].y * height
                        face_boxes.append([face_x1-10, face_y1-10, face_x2+10, face_y2+10])
                        left_eye_boxes.append([left_eye_inner_x1, left_eye_inner_y1, left_eye_inner_x2, left_eye_inner_y2])
                        right_eye_boxes.append([right_eye_inner_x1, right_eye_inner_y1, right_eye_inner_x2, right_eye_inner_y2])

                    face_boxes = np.array(face_boxes)
                    left_eye_boxes = np.array(left_eye_boxes)
                    right_eye_boxes = np.array(right_eye_boxes)

                    # raw copy for reconstruction
                    feed = frame.copy()
                    
                    # Estimate head pose
                    if head_pose_estimation:
                        for results in fa.get_landmarks(feed, face_boxes):
                            pitch, yaw, roll = handler(frame, results, color)

                    # Estimate gaze
                    if gaze_estimation:
                        frame = estimate_gaze_from_face_image(feed, frame, face_boxes, left_eye_boxes, right_eye_boxes, visualization)

                if body_pose_estimation:
                    # Estimate body pose
                    results = pose.process(frame)
                    if results.pose_landmarks:
                        body_landmarks= results.pose_landmarks
                        body_landmarks = np.array([[lmk.x * width, lmk.y * height, lmk.z * width]
                            for lmk in body_landmarks.landmark], dtype=np.float32)
                        
                        # Calculate down-side body pose
                        left_hip = body_landmarks[landmark_names.index('left_hip')]
                        right_hip = body_landmarks[landmark_names.index('right_hip')]

                        # Calculate up-side body pose
                        left_shoulder = body_landmarks[landmark_names.index('left_shoulder')]
                        right_shoulder = body_landmarks[landmark_names.index('right_shoulder')]

                        # Change z-position from the Depth image because the original z-position is estimated position from face pose 
                        # offset is the margin of shoulder position
                        left_y_offset = 10
                        left_x_offset = 20
                        right_x_offset = 20
                        right_y_offset = 10
                        left_shoulder[2] = depth[min(int(left_shoulder[1])+left_y_offset, height-1), min(width-1, int(left_shoulder[0])-left_x_offset)]
                        right_shoulder[2] = depth[min(int(right_shoulder[1])+right_y_offset, height-1), max(0, int(right_shoulder[0])+right_x_offset)]
                        left_hip[2] = depth[min(int(left_hip[1])+left_y_offset, height-1), min(width-1, int(left_hip[0])-left_x_offset)]
                        right_hip[2] = depth[min(int(right_hip[1])+right_y_offset, height-1), max(0, int(right_hip[0])+right_x_offset)]
                        center_hip = (left_hip + right_hip) / 2
                        center_stomach = [int(max(0, min(center_hip[0], width-1))),int(max(0, min((center_hip[1] * 2 + (left_shoulder[1] + right_shoulder[1]))/3, height-1))), 0]
                        center_stomach[2] = depth[center_stomach[1], center_stomach[0]]

                # Visualization
                if visualization:
                    # apply colormap to depthmap
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                    if body_pose_estimation:
                        cv2.circle(depth_colormap, (int(left_shoulder[0]-left_y_offset), int(left_shoulder[1]+left_x_offset)), 3, (0, 255, 0), 3)
                        cv2.circle(depth_colormap, (int(right_shoulder[0]+right_y_offset), int(right_shoulder[1]+right_x_offset)), 3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(left_shoulder[0]-left_y_offset), int(left_shoulder[1]+left_x_offset)), 3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(right_shoulder[0]+right_y_offset), int(right_shoulder[1]+right_x_offset)), 3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(center_stomach[0]), int(center_stomach[1])), 3, (0, 255, 0), 3)
                        cv2.imshow('depth', depth_colormap)
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        if left_shoulder is not None and right_shoulder is not None:
                            upper_body_yaw, upper_body_pitch, upper_body_roll = upside_body_pose_calculator(left_shoulder, right_shoulder, center_stomach)
                            if upper_body_yaw:
                                upper_body_yaw = upper_body_yaw * 180 / math.pi
                                print('yaw_theta', str(upper_body_yaw))
                                upper_body_pitch = upper_body_pitch * 180 / math.pi
                                print('pitch_theta', str(upper_body_pitch))
                                upper_body_roll = upper_body_roll * 180 / math.pi
                                print('roll_theta', str(upper_body_roll))

                    if head_pose_estimation:
                        print('head pose yaw: ', yaw)
                        print('head pose pitch: ', pitch)
                        print('head pose roll: ', roll)

                    if gaze_estimation:
                        for i in range(len(left_eye_boxes)):
                            cv2.rectangle(frame, (int(left_eye_boxes[i][0]), int(left_eye_boxes[i][1])), (int(left_eye_boxes[i][2]), int(left_eye_boxes[i][3])), (255, 255, 0), 1)
                            cv2.rectangle(frame, (int(right_eye_boxes[i][0]), int(right_eye_boxes[i][1])), (int(right_eye_boxes[i][2]), int(right_eye_boxes[i][3])), (0, 255, 0), 1)
                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('MediaPipe Pose', frame)

                    # Check the FPS
                    print('fps = ', 1/(time.time() - start_time))
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
            #except Exception as e:
            #    print('Error is occurred')
            #    print(e)
            #    pass

if __name__ == "__main__":
    main()
