#!/usr/bin/python3
# -*- coding:utf-8 -*-
##############################################################
################## 2022.01.09.ChanHyukLee ####################
##############################################################
# Facial & Body landmark is from mediaPipe (Google)
# Gaze estimation module is from david-wb (https://github.com/david-wb/gaze-estimation)
# Head pose estimation module is from 1996scarlet (https://github.com/1996scarlet/Dense-Head-Pose-Estimation)

import argparse
from cgitb import text
import math
from ssl import ALERT_DESCRIPTION_NO_RENEGOTIATION
from turtle import right
import cv2
import time
import os
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import re

from utils.draw_utils import draw_axis
import utils.visualization_tool as visualization_tool
from Stabilizer.stabilizer import Stabilizer
from utils.inference_module import inference
import head_pose_estimation_module.service as service
from gaze_estimation_module.util.gaze import draw_gaze
from gaze_estimation_module.gaze_estimation import estimate_gaze_from_face_image
import head_pose_estimatior
import body_pose_estimatior
from utils import preprocessing


only_detection_mode = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


use_realsense = False
use_video = False
annotation = False
visualization = False
text_visualization = False
body_pose_estimation = False
head_pose_estimation = True # 12 프레임 저하
gaze_estimation = False # 22프레임 저하
inference_mode = True

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

def main(color=(224, 255, 255), video_path = 'save.avi', save_path = 'data'):
    base_path = os.getcwd()
    write_count = 0
    if annotation:
        body_poses = []
        head_poses = []
        gaze_poses = []
    
    if inference_mode:
        body_poses = []
        head_poses = []
        gaze_poses = []
    
    # Initialization step
    fa = service.DepthFacialLandmarks(os.path.join(base_path, "code/head_pose_estimation_module/weights/sparse_face.tflite"))
    print('Head detection module is initialized')

    # Initialinze Head pose estimation object handler
    handler = getattr(service, 'pose')
    print('Head pose module is initialized')

    if use_realsense:
        align_to = rs.stream.color
        align = rs.align(align_to)

    # Kalman filter initialization

    head_pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    ) for i in range(3)]
    body_pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    ) for i in range(3)]
    left_eye_pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    ) for i in range(2)]
    right_eye_pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    ) for i in range(2)]

    # Define pose estimation & face detection thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
            #try
            print('Camera settings is started')
            # Create a context object. This object owns the handles to all connected realsense devices
            if use_realsense:
                pipeline = rs.pipeline()

                # Configure streams
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                # Start streaming
                pipeline.start(config)
            elif use_video:
                cap = cv2.VideoCapture(video_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print('Video length is ' + str(length))
            else:
                cap = cv2.VideoCapture(4)

            print('Camera settings is initialized')

            # Load the frame from webcam
            while True:
                start_time = time.time()
                #a = cv2.waitKey(10)
                if use_realsense:
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
                else:
                    ret, frame = cap.read()
                    if frame is None:
                        break
                    depth = np.zeros((frame.shape[0], frame.shape[1]))
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                if depth.shape != frame.shape:
                    frame = cv2.resize(frame, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

                # frame shape for return normalized bounding box info
                height, width = frame.shape[:2]

                # Media pipe face detection
                bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(bgr_image)

                # Multi-face detection
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_boxes, left_eye_boxes, right_eye_boxes = head_pose_estimatior.box_extraction(
                            face_landmarks=face_landmarks,
                            width = width,
                            height = height)
                    face_boxes = np.array(face_boxes)
                    left_eye_boxes = np.array(left_eye_boxes)
                    right_eye_boxes = np.array(right_eye_boxes)

                    # raw copy for reconstruction
                    feed = frame.copy()
                    
                    # Estimate head pose
                    if head_pose_estimation:
                        for results in fa.get_landmarks(feed, face_boxes):
                            temp_pitch, temp_yaw, temp_roll = handler(frame, results, color)
                            head_pose_stabilizers[0].update([temp_pitch])
                            head_pose_stabilizers[1].update([temp_yaw])
                            head_pose_stabilizers[2].update([temp_roll])

                            pitch = head_pose_stabilizers[0].state[0][0]
                            yaw = head_pose_stabilizers[1].state[0][0]
                            roll = head_pose_stabilizers[2].state[0][0]

                            if annotation:
                                head_poses.append([yaw, pitch, roll])
                            if inference_mode:
                                head_poses.append([yaw, pitch, roll])
                            
                    # Estimate gaze
                    if gaze_estimation:
                        frame, eyes = estimate_gaze_from_face_image(feed, frame, face_boxes, left_eye_boxes, right_eye_boxes, visualization)
                        left_eye, right_eye = eyes
                        left_gaze = left_eye.gaze.copy()
                        left_gaze[1] = -left_gaze[1]
                        right_gaze = right_eye.gaze.copy()
                        gazes = [left_gaze, right_gaze]
                        right_dx = np.sin(left_gaze[0])
                        left_dx = np.sin(right_gaze[0])

                        left_eye_pose_stabilizers[0].update([gazes[0][0]]) # left eye y coordinate
                        left_eye_pose_stabilizers[1].update([gazes[0][1]]) # left eye x coordinate
                        right_eye_pose_stabilizers[0].update([gazes[1][0]]) # right eye y coordinate
                        right_eye_pose_stabilizers[1].update([gazes[1][1]]) # right eye x coordinate
                        left_x = left_eye_pose_stabilizers[0].state[0][0]
                        left_y = left_eye_pose_stabilizers[1].state[0][0]
                        right_x = right_eye_pose_stabilizers[0].state[0][0]
                        right_y = right_eye_pose_stabilizers[1].state[0][0]
                        gazes = [[left_x, left_y], [right_x, right_y]]
                        if annotation:
                            gaze_poses.append([gazes[0][0], gazes[0][1], gazes[1][0], gazes[1][1]])
                        if inference_mode:
                            gaze_poses.append([gazes[0][0], gazes[0][1], gazes[1][0], gazes[1][1]])
                else:
                    continue

                if body_pose_estimation:
                    # Estimate body pose
                    results = pose.process(frame)
                    if results.pose_landmarks:
                        body_landmarks= results.pose_landmarks
                        body_landmarks = np.array([[lmk.x * width, lmk.y * height, lmk.z * width]
                            for lmk in body_landmarks.landmark], dtype=np.float32)
                        
                        left_shoulder, right_shoulder, center_stomach, left_x_offset, left_y_offset, right_x_offset, right_y_offset = body_pose_estimatior.body_keypoint_extractor(body_landmarks, landmark_names, depth, width, height)
                        upper_body_yaw, upper_body_pitch, upper_body_roll = body_pose_estimatior.upside_body_pose_calculator(left_shoulder, right_shoulder, center_stomach)
                        if upper_body_yaw:
                            temp_upper_body_yaw = upper_body_yaw * 180 / math.pi
                            temp_upper_body_pitch = upper_body_pitch * 180 / math.pi
                            temp_upper_body_roll = upper_body_roll * 180 / math.pi

                            body_pose_stabilizers[0].update([temp_upper_body_pitch])
                            body_pose_stabilizers[1].update([temp_upper_body_yaw])
                            body_pose_stabilizers[2].update([temp_upper_body_roll])

                            upper_body_pitch = body_pose_stabilizers[0].state[0][0]
                            upper_body_yaw = body_pose_stabilizers[1].state[0][0]
                            upper_body_roll = body_pose_stabilizers[2].state[0][0]

                            if annotation:
                                body_poses.append([upper_body_yaw, upper_body_pitch, upper_body_roll])
                            if inference_mode:
                                body_poses.append([upper_body_yaw, upper_body_pitch, upper_body_roll])

                # Visualization
                if visualization:
                    # apply colormap to depthmap
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                    if body_pose_estimation and results.pose_landmarks:
                        cv2.circle(depth_colormap, (int(left_shoulder[0]-left_y_offset), int(left_shoulder[1]+left_x_offset)), 3, (0, 255, 0), 3)
                        cv2.circle(depth_colormap, (int(right_shoulder[0]+right_y_offset), int(right_shoulder[1]+right_x_offset)), 3, (0, 255, 0), 3)
                        cv2.imshow('depth', depth_colormap)
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        if left_shoulder is not None and right_shoulder is not None:
                            if upper_body_yaw:
                                frame = draw_axis(frame, upper_body_yaw, upper_body_pitch, upper_body_roll, [int((left_shoulder[0] + right_shoulder[0])/2), int(left_shoulder[1])],
                                color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))

                    if head_pose_estimation and yaw:
                        #print('head pose yaw: ', yaw)
                        #print('head pose pitch: ', pitch)
                        #print('head pose roll: ', roll)
                        frame = draw_axis(frame, yaw, pitch, roll, [int((face_boxes[0][0] + face_boxes[0][2])/2), int(face_boxes[0][1] - 30)])
                    
                    if gaze_estimation:
                        for i, ep in enumerate([left_eye, right_eye]):
                            for (x, y) in ep.landmarks[16:33]:
                                color = (0, 255, 0)
                                if ep.eye_sample.is_left:
                                    color = (255, 0, 0)
                                cv2.circle(frame,
                                            (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
                            gaze = gazes[i]
                            length = 60.0
                            draw_gaze(frame, ep.landmarks[-2], gaze, length=length, thickness=2)

                    if text_visualization:
                        width = int(width * 1.5)
                        zero_array = np.zeros((height, width, 3), dtype=np.uint8)
                        if body_pose_estimation and results.pose_landmarks:
                            center_shoulder = (left_shoulder + right_shoulder) / 2
                            zero_array= visualization_tool.draw_body_information(zero_array, width, height, round(center_shoulder[0], 2), round(center_shoulder[1], 2), round(center_shoulder[2], 2), 
                            round(upper_body_yaw, 2), round(upper_body_pitch, 2), round(upper_body_roll, 2))
                        if head_pose_estimation and yaw:
                            face_center_x = (face_boxes[0][0] + face_boxes[0][2]) / 2
                            face_center_y = (face_boxes[0][1] + face_boxes[0][3]) / 2
                            if len(depth.shape) > 2:
                                face_center_z = depth[int(face_center_y), int(face_center_x), 0]
                            else:
                                face_center_z = depth[int(face_center_y), int(face_center_x)]
                            zero_array = visualization_tool.draw_face_information(zero_array, width, height, round(face_center_x, 2), round(face_center_y, 2), round(face_center_z, 2), round(yaw, 2)
                            , round(pitch, 2), round(roll, 2))
                        if gaze_estimation and gazes is not None:
                            center_eye_x = (left_eye_boxes[0][0] + left_eye_boxes[0][2]) / 2
                            center_eye_y = (left_eye_boxes[0][1] + left_eye_boxes[0][3]) / 2
                            center_eye_z = depth[int(center_eye_y), int(center_eye_x)]
                            zero_array = visualization_tool.draw_gaze_information(zero_array,width, height, round(center_eye_x, 2), round(center_eye_y, 2), round(center_eye_z, 2), gazes)
                        stacked_frame = np.concatenate([frame, zero_array], axis=1)


                    # Flip the image horizontally for a selfie-view display.
                    #cv2.imshow('MediaPipe Pose2', stacked_frame)

                if head_pose_estimation and (inference_mode):
                    if len(head_poses) >=58:
                        head_poses = head_poses[-58:]
                        data = preprocessing.data_preprocessing(np.array(head_poses))
                        inputs = np.expand_dims(np.array(data),axis=0)
                        human_state = inference(inputs)
                        cv2.putText(frame, human_state, (100, 100), 1, 2, (0, 0, 0), 3)
                cv2.imshow('MediaPipe Pose1', frame)
                    # Check the FPS
                #print('fps = ', 1/(time.time() - start_time))
                pressed_key = cv2.waitKey(1)
                if pressed_key == 27:
                    break

    if annotation:
        if head_pose_estimation:
            head_poses = np.array(head_poses)
        if body_pose_estimation:
            body_poses = np.array(body_poses)
        if gaze_estimation:
            gaze_poses = np.array(gaze_poses)
        if not os.path.exists(os.path.join(base_path, save_path)):
            os.mkdir(os.path.join(base_path, save_path))
        video_name = video_path.split('/')[-1]
        numbers = re.sub(r'[^0-9]', '', video_name)
        np.save(os.path.join(base_path, save_path, numbers), head_poses)

if __name__ == "__main__":
    if annotation:
        base_path = os.getcwd()
        actions = ['standard', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+']
        for index, action in enumerate(actions):
            video_folder_path = os.path.join(base_path, 'dataset', 'test_video', str(index) + '.' + action)
            for video in os.listdir(video_folder_path):
                main(
                    video_path = os.path.join(video_folder_path, video),
                    save_path = 'dataset/test_npy/' + str(index) + '.' + action
                    )
    elif inference_mode:
        if not use_video:
            main()
        else:
            base_path = os.getcwd()
            actions = ['standard', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+']
            for index, action in enumerate(actions):
                video_folder_path = os.path.join(base_path, 'dataset', 'test_video', str(index) + '.' + action)
                for video in os.listdir(video_folder_path):
                    main(
                        video_path = os.path.join(video_folder_path, video),
                    save_path = 'dataset/test_npy/' + str(index) + '.' + action
                        )
    else:
        main()