#!/usr/bin/python3
# -*- coding:utf-8 -*-
##############################################################
################## 2022.01.09.ChanHyukLee ####################
##############################################################

import argparse
import math
from turtle import right
from gaze_estimation_module.gaze_estimation import estimate_gaze_from_face_image
import head_pose_estimation_module.service as service
import cv2
import time
import os
import mediapipe as mp
import numpy as np

only_detection_mode = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection


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


def upside_body_pose_calculator(left_shoulder, right_shoulder):
    center_shoulder = (left_shoulder + right_shoulder) / 2
    # Yaw
    if left_shoulder[2] > right_shoulder[2]: # yaw (-) direction
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [-1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        return theta * -1
    else:
        direction_vector = (left_shoulder - center_shoulder)
        direction_vector = (direction_vector[0], direction_vector[2])
        pivot_vector = [1, 0]
        theta = (direction_vector[0]*pivot_vector[1] - direction_vector[1] * pivot_vector[0]) / ((math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)) * (math.sqrt(pivot_vector[0] ** 2 + pivot_vector[1] ** 2)))
        theta = math.asin(theta)
        return theta


def main(color=(224, 255, 255)):
    base_path = os.getcwd()

    # Tensorflow lite detection model. (Deep learning based. But slower than mediapipe's face detection model)
    #fd = service.UltraLightFaceDetecion(os.path.join(base_path,"human_pose/head_pose_estimation_module/weights/RFB-320.tflite"),
    #                                    conf_threshold=0.95)

    fa = service.DepthFacialLandmarks(os.path.join(base_path, "head_pose_estimation_module/weights/sparse_face.tflite"))

    # Initialinze Head pose estimation object handler
    handler = getattr(service, 'pose')
    cap = cv2.VideoCapture(0)

    # Define pose estimation & face detection thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened:
                # Load the frame from webcam
                start_time = time.time()
                ret, frame = cap.read()

                # frame shape for return normalized bounding box info
                height, width = frame.shape[:2]

                # Inference time check

                if not ret:
                    break

                # face detection from other deep learning model. (ECCV, 2020)
                #boxes, scores = fd.inference(frame)

                # Media pipe face detection
                results = face_detection.process(frame)
                boxes = []

                # Multi-face detection
                if results.detections:
                    for detection in results.detections:
                        box_x_min = detection.location_data.relative_bounding_box.xmin * width
                        box_y_min = detection.location_data.relative_bounding_box.ymin * height
                        box_width = detection.location_data.relative_bounding_box.width * width
                        box_height = detection.location_data.relative_bounding_box.height * height
                        relative_keypoints = detection.location_data.relative_keypoints
                        right_eye_x = relative_keypoints[0].x * width
                        right_eye_y = relative_keypoints[0].y * height
                        left_eye_x = relative_keypoints[1].x * width
                        left_eye_y = relative_keypoints[1].y * height
                        boxes.append([box_x_min, box_y_min, box_x_min + box_width - 1, box_y_min + box_height - 1])
                    boxes = np.array(boxes)

                    # raw copy for reconstruction
                    feed = frame.copy()
                    
                    # Estimate head pose
                    if not only_detection_mode:
                        for results in fa.get_landmarks(feed, boxes):
                            pitch, yaw, roll = handler(frame, results, color)
                        #print('pitch = ', pitch, 'yaw = ', yaw, 'roll = ',roll)
                        # cv2.imwrite(f'draft/gif/trans/img{counter:0>4}.jpg', frame)

                        frame.flags.writeable = False
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Estimate body pose
                        results = pose.process(frame)
                        if results.pose_landmarks:
                            body_landmarks= results.pose_landmarks
                            body_landmarks = np.array([[lmk.x * width, lmk.y * height, lmk.z * width]
                                for lmk in body_landmarks.landmark], dtype=np.float32)
                            left_hip = body_landmarks[landmark_names.index('left_hip')]
                            right_hip = body_landmarks[landmark_names.index('right_hip')]
                            center_hip = (left_hip + right_hip) / 2
                            left_knee = body_landmarks[landmark_names.index('left_knee')]
                            right_knee = body_landmarks[landmark_names.index('right_knee')]
                            center_knee = (left_knee + right_knee) / 2
                            downside_body_vector = center_hip - center_knee
                            left_shoulder = body_landmarks[landmark_names.index('left_shoulder')]
                            right_shoulder = body_landmarks[landmark_names.index('right_shoulder')]
                            print('Left: ', str(left_shoulder))
                            print('Right: ', str(right_shoulder))
                            if left_shoulder is not None and right_shoulder is not None:
                                theta = upside_body_pose_calculator(left_shoulder, right_shoulder)
                                if theta:
                                    theta = theta * 180 / math.pi
                                    print('yaw_theta', str(theta))
                        

                        # Draw the pose annotation on the image.
                        frame.flags.writeable = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # For visualization
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                        # For gaze estimation
                        face_image = frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]
                        frame = estimate_gaze_from_face_image(feed,frame, face_image)
                        # Flip the image horizontally for a selfie-view display.
                        cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))

                        # Check the FPS
                    #print('fps = ', 1/(time.time() - start_time))
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

if __name__ == "__main__":
    main()
