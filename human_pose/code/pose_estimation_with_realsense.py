#!/usr/bin/python3
# -*- coding:utf-8 -*-
##############################################################
################## 2022.01.09.ChanHyukLee ####################
##############################################################

import argparse
import math
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
    print('Head detection module is initialized')

    # Initialinze Head pose estimation object handler
    handler = getattr(service, 'pose')
    print('Head pose module is initialized')

    # Define pose estimation & face detection thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            try:
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
                    frame = frames.get_color_frame()
                    depth = frames.get_depth_frame()
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
                            print('pitch = ', pitch, 'yaw = ', yaw, 'roll = ',roll)
                            # cv2.imwrite(f'draft/gif/trans/img{counter:0>4}.jpg', frame)

                            frame.flags.writeable = False
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # Estimate body pose
                            results = pose.process(frame)
                            if results.pose_landmarks:
                                body_landmarks= results.pose_landmarks
                                body_landmarks = np.array([[lmk.x * width, lmk.y * height, lmk.z * width]
                                    for lmk in body_landmarks.landmark], dtype=np.float32)
                                
                                # Calculate down-side body pose
                                left_hip = body_landmarks[landmark_names.index('left_hip')]
                                right_hip = body_landmarks[landmark_names.index('right_hip')]
                                center_hip = (left_hip + right_hip) / 2
                                left_knee = body_landmarks[landmark_names.index('left_knee')]
                                right_knee = body_landmarks[landmark_names.index('right_knee')]
                                center_knee = (left_knee + right_knee) / 2
                                downside_body_vector = center_hip - center_knee

                                # Calculate up-side body pose
                                left_shoulder = body_landmarks[landmark_names.index('left_shoulder')]
                                right_shoulder = body_landmarks[landmark_names.index('right_shoulder')]

                                # Change z-position from the Depth image because the original z-position is estimated position from face pose 
                                # 30 is the margin of shoulder position
                                left_y_offset = 10
                                left_x_offset = 20
                                right_x_offset = 20
                                right_y_offset = 10
                                left_shoulder[2] = depth[int(left_shoulder[1])+left_y_offset, int(left_shoulder[0])-left_x_offset]
                                right_shoulder[2] = depth[int(right_shoulder[1])+right_y_offset, int(right_shoulder[0])+right_x_offset]
                                cv2.circle(frame, (int(left_shoulder[0]-left_y_offset), int(left_shoulder[1]+left_x_offset)), 3, (0, 255, 0), 3)
                                cv2.circle(frame, (int(right_shoulder[0]+right_y_offset), int(right_shoulder[1]+right_x_offset)), 3, (0, 255, 0), 3)
                                #print(left_shoulder)
                                print('Left: ', str(left_shoulder))
                                print('Right: ', str(right_shoulder))

                                # Visualization
                                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                                cv2.circle(depth_colormap, (int(left_shoulder[0]-left_y_offset), int(left_shoulder[1]+left_x_offset)), 3, (0, 255, 0), 3)
                                cv2.circle(depth_colormap, (int(right_shoulder[0]+right_y_offset), int(right_shoulder[1]+right_x_offset)), 3, (0, 255, 0), 3)
                                cv2.imshow('depth', depth_colormap)
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
                            box = boxes[0]
                            #face_image = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                            face_coordinate = np.expand_dims(np.array([int(box[1]),int(box[0]), int(box[2] - box[0]), int(box[3] - box[1])]), axis=0)
                            print(face_coordinate)
                            frame = estimate_gaze_from_face_image(feed, frame, face_coordinate)

                            # Flip the image horizontally for a selfie-view display.
                            cv2.imshow('MediaPipe Pose', frame)

                            # Check the FPS
                        print('fps = ', 1/(time.time() - start_time))
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
            except Exception as e:
                print('Error is occurred')
                print(e)
                pass

if __name__ == "__main__":
    main()
