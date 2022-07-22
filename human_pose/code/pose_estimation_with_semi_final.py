#!/usr/bin/python3
# -*- coding:utf-8 -*-
##############################################################
################## 2022.04.02.ChanHyukLee ####################
##############################################################
# Facial & Body landmark is from mediaPipe (Google)
# Gaze estimation module is from david-wb (https://github.com/david-wb/gaze-estimation)
# Head pose estimation module is from 1996scarlet (https://github.com/1996scarlet/Dense-Head-Pose-Estimation)

import argparse
from cgitb import text
from inspect import stack
from lib2to3.pytree import BasePattern
import math
from ssl import ALERT_DESCRIPTION_NO_RENEGOTIATION
from sys import flags
from turtle import right
import cv2
import time
import os
from matplotlib.pyplot import fill
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import re
import random
from torch import _empty_affine_quantized
import zmq
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Don't change
inference_mode = True
inference_mode_in_the_wild = True

only_detection_mode = False
use_realsense = True
use_video = False
annotation = False
visualization = True
text_visualization = False
body_pose_estimation = True
head_pose_estimation = True # 12 프레임 저하
gaze_estimation = False # 22프레임 저하
result_record = False
zmq_enable = True
only_action_mode = False

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

actions = [ 'left', 'left-up', 'up',
'right-up', 'right', 'right-down', 'down', 'left-down', 'zoom-in', 'zoom-out','standard']

def fill_the_blank(poses):
    if len(poses) > 0:
        poses.append(poses[-1])
    else:
        poses.append([1.0, 1.0, 1.0, 1.0])
    return poses

def fps_quantization(fps):
    if fps < 15:
        fps = 10
    elif fps < 20:
        fps = 15
    elif fps < 25:
        fps = 20
    elif fps < 30:
        fps  = 25
    elif fps < 35:
        fps = 30
    elif fps < 40:
        fps = 35
    else:
        fps = 40
    return fps

def calibration(center_eyes):
    # for D435
    camera_horizontal_angle = 85.2
    camera_vertical_angle = 58

    i_width = 640
    i_height = 480
    
    # before calibration
    eye_x = (i_width/2) - center_eyes[0]
    eye_y = (i_height/2) - center_eyes[1]
    eye_z = center_eyes[2]

    detected_x_angle = (camera_horizontal_angle / 2) * (eye_x / (i_width/2))
    detected_y_angle = (camera_vertical_angle / 2) * (eye_y / (i_height/2))

    new_x = eye_z * math.sin(math.radians(detected_x_angle)) * 7 / 9
    new_y = eye_z * math.sin(math.radians(detected_y_angle))
    y_offset = eye_z * math.sin(math.radians(camera_vertical_angle/2))

    return new_x, new_y + 240, eye_z

def main(color=(224, 255, 255), rgb_video_path = 'save.avi', depth_video_path = 'save.avi', save_path = 'data'):
    base_path = os.getcwd()
    fps = 20
    order = 0
    last_action=0
    
    if annotation:
        body_poses = []
        head_poses = []
        gaze_poses = []
        center_eyes = []
        center_mouths = []
        left_shoulders = []
        right_shoulders = []
        center_stomachs = []
    
    if inference_mode:
        body_poses = []
        head_poses = []
        gaze_poses = []
        center_eyes = []
        center_mouths = []
        left_shoulders = []
        right_shoulders = []
        center_stomachs = []
        human_state = None
        for i in range(200):
            fill_the_blank(head_poses)
            fill_the_blank(center_eyes)
            fill_the_blank(center_mouths)
            fill_the_blank(left_shoulders)
            fill_the_blank(right_shoulders)
            fill_the_blank(center_stomachs)
    
    yaw, pitch, roll = 0, 0, 0
    
    # Initialization step
    fa = service.DepthFacialLandmarks("C:/Users/user/Desktop/version/human_pose_estimation/human_pose/code/head_pose_estimation_module/weights/sparse_face.tflite")
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
            max_num_faces=3,
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
                rgb_cap = cv2.VideoCapture(rgb_video_path)
                depth_cap = cv2.VideoCapture(depth_video_path)
                length = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print('Video length is ' + str(length))
            else:
                cap = cv2.VideoCapture(0)

            if result_record:
                w = round(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = round(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = rgb_cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

                # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                rgb_video_name = rgb_video_path.split('/')[-1]
                rgb_video_name = rgb_video_name[0:len(rgb_video_name)-4] + '.avi'
                original_out = cv2.VideoWriter(
                    os.path.join(save_path, rgb_video_name)
                    , fourcc, fps, (1600, 480))
                result_out = cv2.VideoWriter(
                    os.path.join(save_path, rgb_video_name)
                    , fourcc, fps, (1600, 480))

            print('Camera settings is initialized')

            # Load the frame from webcam
            print("Please turn on the super multiview renderer")
            while True:
                # Communication settings
                communication_read = open('communication.txt', 'r')
                line = ['0', '0 0 0', '0 0 0', 'standard'] # tracking_mode / eye_position / head_rotation / human action
                for i in range(4): # mode / eye_position / head_rotation / action
                    line[i] = communication_read.readline()
                if line[0].strip() == '0':
                    communication_read.close()
                    continue
                elif line[0].strip() == '1':
                    only_detection_mode = True
                    head_pose_estimation = False
                    body_pose_estimation = False
                elif line[0].strip() == '2':
                    only_detection_mode = False
                    head_pose_estimation = True
                    body_pose_estimation = False
                elif line[0].strip() == '3':
                    only_detection_mode = False
                    head_pose_estimation = True
                    body_pose_estimation = True
                else:
                    print('Undefined mode is occurred. Please modify the line 1 in the communication.txt file')
                communication_read.close()

                start_time = time.time()
                face_box_per_man = []
                head_pose_per_man = []
                body_pose_per_man = []
                eye_pose_per_man = []
                depth_per_man = []
                center_face_per_man = []
                no_looking = []
                all_estimation_is_successed = True
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
                elif use_video:
                    ret, frame = rgb_cap.read()
                    ret, depth = depth_cap.read()
                    if frame is None:
                        break
                else:
                    ret, frame = cap.read()
                    if frame is None:
                        break
                    depth = np.zeros((frame.shape[0], frame.shape[1]))
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                frame_copy = frame.copy()
                frame_copy = cv2.flip(frame_copy, 1)
                cv2.imshow('frame', frame_copy)
                cv2.waitKey(1)
                if depth.shape != frame.shape:
                    frame = cv2.resize(frame, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

                #depth = depth / 20

                # frame shape for return normalized bounding box info
                height, width = frame.shape[:2]

                # Media pipe face detection
                bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(bgr_image)

                # Multi-face detection
                if results.multi_face_landmarks:
                    for index, face_landmarks in enumerate(results.multi_face_landmarks):
                        face_boxes, left_eye_boxes, right_eye_boxes = head_pose_estimatior.box_extraction(
                            face_landmarks=face_landmarks,
                            width = width,
                            height = height)
                        face_boxes = np.array(face_boxes)
                        face_box_per_man.append(face_boxes)
                        center_face_x = int((face_boxes[0][0] + face_boxes[0][2]) / 2)
                        center_face_y = int(face_boxes[0][3])
                        center_depth_of_face = depth[center_face_y, center_face_x]
                        depth_per_man.append(center_depth_of_face)
                        center_face_per_man.append((center_face_x, center_face_y))
                        left_eye_boxes = np.array(left_eye_boxes)
                        right_eye_boxes = np.array(right_eye_boxes)
                        center_eye_x = (left_eye_boxes[0][0] + left_eye_boxes[0][2]) / 2
                        center_eye_y = (left_eye_boxes[0][1] + left_eye_boxes[0][3]) / 2
                        if len(depth.shape) > 2:
                            center_eye_z = depth[max(0, min(479, int(center_eye_y))), max(0, min(639, int(center_eye_x))), 0]
                        else:
                            center_eye_z = depth[max(0, min(479, int(center_eye_y))), max(0, min(639, int(center_eye_x)))]
                        center_eye = [center_eye_x, center_eye_y, center_eye_z, 0]
                        center_eyes.append(center_eye)
                        #cv2.rectangle(frame, (int(left_eye_boxes[0][0]), int(left_eye_boxes[0][1])), (int(left_eye_boxes[0][2]), int(left_eye_boxes[0][3])), (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('frame3', frame)
                        cv2.waitKey(1)


                    # raw copy for reconstruction
                    feed = frame.copy()
                    
                    # Estimate head pose
                    if head_pose_estimation:
                        for face_boxes in face_box_per_man:
                            for results in fa.get_landmarks(feed, face_boxes):
                                pitch, yaw, roll = handler(frame, results, color)
                                #head_pose_stabilizers[0].update([temp_pitch])
                                #head_pose_stabilizers[1].update([temp_yaw])
                                #head_pose_stabilizers[2].update([temp_roll])

                                #pitch = head_pose_stabilizers[0].state[0][0]
                                #yaw = head_pose_stabilizers[1].state[0][0]
                                #roll = head_pose_stabilizers[2].state[0][0]

                                if inference_mode:
                                    head_pose_per_man.append([yaw, pitch, roll, 0])
                                    head_poses.append([yaw, pitch, roll, 0])
                        
                    min_val = min(depth_per_man)
                    min_index = depth_per_man.index(min_val)
                    min_head_pose = 999
                    min_head_pose_index = 0
                    for index, face in enumerate(face_box_per_man):
                        if index == min_index:
                            cv2.line(frame, (int(width/2), height-1), center_face_per_man[index], (0, 255, 0), 3)
                            cv2.rectangle(frame, (int(face_box_per_man[index][0][0]), int(face_box_per_man[index][0][1])), (int(face_box_per_man[index][0][2]), int(face_box_per_man[index][0][3])), (0,255,0), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'Main User', (int(face[0][0]), int(face[0][1])), 1, 2, (0, 255, 0), 2)
                        else:
                            cv2.line(frame, (int(width/2), height-1), center_face_per_man[index], (255, 0, 0), 1)
                            cv2.rectangle(frame, (int(face_box_per_man[index][0][0]), int(face_box_per_man[index][0][1])), (int(face_box_per_man[index][0][2]), int(face_box_per_man[index][0][3])), (255,0,0), 2, cv2.LINE_AA)
                            
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
                            eye_pose_per_man.append([gazes[0][0], gazes[0][1], gazes[1][0], gazes[1][1]])
                            gaze_poses.append([gazes[0][0], gazes[0][1], gazes[1][0], gazes[1][1]])
                else:
                    if annotation or inference_mode:
                        head_poses = fill_the_blank(head_poses)
                        gaze_poses = fill_the_blank(gaze_poses)
                        center_eyes = fill_the_blank(center_eyes)
                        all_estimation_is_successed = False

                if body_pose_estimation:
                    # Estimate body pose
                    results = pose.process(frame)
                    if results.pose_landmarks:
                        body_landmarks= results.pose_landmarks
                        body_landmarks = np.array([[lmk.x * width, lmk.y * height, lmk.z * width]
                            for lmk in body_landmarks.landmark], dtype=np.float32)
                        
                        if use_video is False and use_realsense is False:
                            normal_state = True
                        else:
                            normal_state = False
                        left_shoulder, right_shoulder, center_stomach, center_mouth, left_x_offset, left_y_offset, right_x_offset, right_y_offset, center_eye3 = body_pose_estimatior.body_keypoint_extractor(body_landmarks, landmark_names, depth, width, height, normal_camera=normal_state, use_realsense = use_realsense)
                        frame = visualization_tool.draw_body_keypoints(frame, [left_shoulder, right_shoulder, center_stomach, center_mouth, center_eye3])
                        upper_body_yaw, upper_body_pitch, upper_body_roll = body_pose_estimatior.upside_body_pose_calculator(left_shoulder, right_shoulder, center_stomach)

                        if annotation or inference_mode:
                            center_stomachs.append(np.append(center_stomach, [0], 0))
                            center_mouths.append(np.append(center_mouth, [0], 0))
                            left_shoulders.append(np.append(left_shoulder, [0], 0))
                            right_shoulders.append(np.append(right_shoulder, [0], 0))

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
                            body_poses.append([upper_body_yaw, upper_body_pitch, upper_body_roll, 0])
                        if inference_mode:
                            body_pose_per_man.append([upper_body_yaw, upper_body_pitch, upper_body_roll])
                            body_poses.append([upper_body_yaw, upper_body_pitch, upper_body_roll, 0])

                    else:
                        if annotation or inference_mode:
                            body_poses = fill_the_blank(body_poses)
                            center_stomachs = fill_the_blank(center_stomachs)
                            center_mouths = fill_the_blank(center_mouths)
                            left_shoulders = fill_the_blank(left_shoulders)
                            right_shoulders = fill_the_blank(right_shoulders)
                            all_estimation_is_successed = False

                # Visualization
                if visualization: #and all_estimation_is_successed:
                    # apply colormap to depthmap
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                    if body_pose_estimation and results.pose_landmarks:
                        cv2.circle(depth_colormap, (int(left_shoulder[0]-left_y_offset), int(left_shoulder[1]+left_x_offset)), 3, (0, 255, 0), 3)
                        cv2.circle(depth_colormap, (int(right_shoulder[0]+right_y_offset), int(right_shoulder[1]+right_x_offset)), 3, (0, 255, 0), 3)
                        cv2.imshow('depth', depth_colormap)
                        #mp_drawing.draw_landmarks(
                        #    frame,
                        #    results.pose_landmarks,
                        #    mp_pose.POSE_CONNECTIONS,
                        #    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        if left_shoulder is not None and right_shoulder is not None:
                            frame = draw_axis(frame, upper_body_yaw, upper_body_pitch, upper_body_roll, [int((left_shoulder[0] + right_shoulder[0])/2), int(left_shoulder[1])],
                            color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))

                    if head_pose_estimation and yaw:
                        #print('head pose yaw: ', yaw)
                        #print('head pose pitch: ', pitch)
                        #print('head pose roll: ', roll)
                        for index, face_boxes in enumerate(face_box_per_man):
                            yaw, pitch, roll = head_pose_per_man[index][0], head_pose_per_man[index][1], head_pose_per_man[index][2]
                            frame = draw_axis(frame, yaw, pitch, roll, [int((face_boxes[0][0] + face_boxes[0][2])/2), int(face_boxes[0][1] - 30)])
                            #if body_pose_estimation is False:
                                #cv2.putText(frame, 'yaw=' + str(int(yaw)), (0, 50), 1, 3, (0, 0, 255), 2)
                                #cv2.putText(frame, 'pitch=' + str(int(pitch)), (0, 100), 1, 3, (0, 0, 255), 2)
                                #cv2.putText(frame, 'roll=' + str(int(roll)), (0, 150), 1, 3, (0, 0, 255), 2)


                    #if only_detection_mode:
                    #    if results.multi_face_landmarks:
                    #        #cv2.rectangle(frame, (int(face_boxes[0][0]), int(face_boxes[0][1])), (int(face_boxes[0][2]), int(face_boxes[0][3])), (255, 0, 0), 2, cv2.LINE_AA)

                    if gaze_estimation:
                        for i, ep in enumerate([left_eye, right_eye]):
                            for (x, y) in ep.landmarks[16:33]:
                                color = (0, 255, 0)
                                if ep.eye_sample.is_left:
                                    color = (255, 0, 0)
                                cv2.circle(frame,(int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
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
                                face_center_z = depth[max(0, min(479, int(face_center_y))), max(0, min(639, int(face_center_x))), 0]
                            else:
                                face_center_z = depth[max(0, min(479,int(face_center_y))), max(0, min(639,int(face_center_x)))]
                            zero_array = visualization_tool.draw_face_information(zero_array, width, height, round(face_center_x, 2), round(face_center_y, 2), round(face_center_z, 2), round(yaw, 2)
                            , round(pitch, 2), round(roll, 2))
                        if gaze_estimation and gazes is not None:
                            zero_array = visualization_tool.draw_gaze_information(zero_array,width, height, round(center_eye_x, 2), round(center_eye_y, 2), round(center_eye_z, 2), gazes)
                        stacked_frame = np.concatenate([frame, zero_array], axis=1)

                    # Flip the image horizontally for a selfie-view display.

                if inference_mode_in_the_wild:
                    rounded_fps = round(fps)
                    center_eyes = center_eyes[len(center_eyes)-200: len(center_eyes)]
                    center_mouths = center_mouths[len(center_mouths)-200: len(center_mouths)]
                    left_shoulders = left_shoulders[len(left_shoulders)-200: len(left_shoulders)]
                    right_shoulders = right_shoulders[len(right_shoulders) - 200 : len(right_shoulders)]
                    center_stomachs = center_stomachs[len(center_stomachs) - 200 : len(center_stomachs)]
                    head_poses = head_poses[len(head_poses) - 200 : len(head_poses)]
                    body_poses = body_poses[len(body_poses) - 200 : len(body_poses)]
                    gaze_poses = gaze_poses[len(gaze_poses) - 200 : len(gaze_poses)] 
                    if head_pose_estimation and body_pose_estimation:
                        if len(head_poses) > rounded_fps*2:
                            temp_center_eyes = np.array(center_eyes[len(center_eyes) - rounded_fps*2:len(center_eyes)])
                            temp_center_mouths = np.array(center_mouths[len(center_mouths) - 2*rounded_fps: len(center_mouths)])
                            temp_left_shoulders = np.array(left_shoulders[len(left_shoulders) - 2*rounded_fps: len(left_shoulders)])
                            temp_right_shoulders = np.array(right_shoulders[len(right_shoulders) - rounded_fps*2 : len(right_shoulders)])
                            temp_center_stomachs = np.array(center_stomachs[len(center_stomachs) - 2*rounded_fps : len(center_stomachs)])
                            temp_head_poses = head_poses[len(head_poses) - 2*rounded_fps : len(head_poses)]
                            temp_body_poses = body_poses[len(body_poses) - 2*rounded_fps : len(body_poses)]
                            temp_gaze_poses = gaze_poses[len(gaze_poses) - 2*rounded_fps : len(gaze_poses)] 
                        if len(temp_head_poses) == rounded_fps*2 and rounded_fps>5:
                            output = [temp_center_eyes, temp_center_mouths, temp_left_shoulders, temp_right_shoulders, temp_center_stomachs]
                            if head_pose_estimation:
                                head_poses_np = np.array(temp_head_poses)
                                output.append(head_poses_np)
                            #if body_pose_estimation:
                            #    body_poses_np = np.array(temp_body_poses)
                            #    output.append(body_poses_np)
                            #if gaze_estimation:
                            #    gaze_poses_np = np.array(temp_gaze_poses)
                            #    output.append(gaze_poses_np)

                            # 총 25개의 features
                            output = np.array(output)
                            while output.shape[1] < rounded_fps*2:
                                output = np.append(output, output[:, -1, :].reshape(output.shape[0], 1, output.shape[2]), axis=1)
                            data = preprocessing.data_preprocessing(output, rounded_fps)
                            inputs = np.expand_dims(np.array(data),axis=0)
                            human_state = inference(inputs)
                            key = cv2.waitKey(1)
                            #if key == ord('9') and time.time() - last_action > 1:
                            #    human_state = actions[order%11]
                            #    order += 1
                            #    last_action = time.time()
                            #else:
                            #    human_state = 'standard'
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, human_state, (0, 50), 1, 3, (0, 0, 255), 3)
                            cv2.imshow('frame3', frame)
                            cv2.waitKey(1)
                    if zmq_enable:
                        if only_action_mode:
                            communication_write = open('communication.txt', 'r+')
                            communication_write.write(line[0])
                            communication_write.write(str(round(0)).zfill(3) + ' ' + str(round(200)).zfill(3) + ' ' + str(round(600)).zfill(3) + '\n')
                            communication_write.write(str(round(head_poses[-1][1])).zfill(3) + ' ' + str(0).zfill(3) + ' ' + str(round(head_poses[-1][2])).zfill(3) + '\n' if head_pose_estimation else '0 0 0\n')
                            communication_write.write(human_state+'\n' if human_state is not None else 'standard\n')
                            communication_write.close()
                        else:
                            if len(depth_per_man) > 0:
                                min_val = min(depth_per_man)
                                min_index = depth_per_man.index(min_val)
                                if min_index < len(head_pose_per_man):
                                    main_center_eye = center_face_per_man[min_index]
                                    main_head_pose = head_pose_per_man[min_index]
                                    eye_x, eye_y, eye_z = calibration([main_center_eye[0], main_center_eye[1], min_val])
                                    communication_write = open('communication.txt', 'r+')
                                    communication_write.write(line[0])
                                    #print('eye position')
                                    #print(eye_x, eye_y, eye_z)
                                    #print('head pose')
                                    #print(head_poses[-1][0], head_poses[-1][1], head_poses[-1][2])
                                    communication_write.write(str(round(eye_x)).zfill(3) + ' ' + str(round(eye_y+20)).zfill(3) + ' ' + str(round(eye_z)).zfill(3) + '\n')
                                    communication_write.write(str(round(main_head_pose[1])).zfill(3) + ' ' + str(round(main_head_pose[0])).zfill(3) + ' ' + str(round(main_head_pose[2])).zfill(3) + '\n' if head_pose_estimation else '0 0 0\n')
                                    communication_write.write(human_state+'\n' if human_state is not None else 'standard\n')
                                    communication_write.close()






                cv2.namedWindow('MediaPipe Pose1', cv2.WINDOW_NORMAL)
                cv2.imshow('MediaPipe Pose1', frame)
                #cv2.imshow('MediaPipe Pose2', stacked_frame)
                if result_record:
                    result_out.write(stacked_frame.copy())

                    # Check the FPS
                fps = 1 / (time.time() - start_time)
                #print(fps)
                pressed_key = cv2.waitKey(1)
                if pressed_key == 27:
                    break

    if annotation:
        output = [center_eyes, center_mouths, left_shoulders, right_shoulders, center_stomachs]
        if head_pose_estimation:
            head_poses = np.array(head_poses)
            output.append(head_poses)
        if body_pose_estimation:
            body_poses = np.array(body_poses)
            output.append(body_poses)
        if gaze_estimation:
            gaze_poses = np.array(gaze_poses)
            output.append(gaze_poses)


        # 총 25개의 features
        output = np.array(output)
        while output.shape[1] < 60:
            output = np.append(output, output[:, -1, :].reshape(output.shape[0], 1, output.shape[2]), axis=1)
        if not os.path.exists(os.path.join(base_path, save_path)):
            os.mkdir(os.path.join(base_path, save_path))
        video_name = rgb_video_path.split('/')[-1]
        numbers = re.sub(r'[^0-9]', '', video_name)
        np.save(os.path.join(base_path, save_path, numbers), output)
    
    if result_record:
        original_out.release()
        result_out.release()

def main_function():
    if annotation:
        base_path = os.getcwd()
        actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+', 'left', 'left_up', 'up',
        'right_up', 'right', 'right_down', 'down', 'left_down', 'zoom_in', 'zoom_out', 'standard']
        for index, action in enumerate(actions):
            video_folder_path = os.path.join(base_path, 'dataset/additional_dataset/other_people/hyemin/hyemin_video_train', str(index) + '.' + action)
            rgb_videos = []
            depth_videos = []
            for video in os.listdir(video_folder_path):
                if 'depth' in video:
                    depth_videos.append(video)
                else:
                    rgb_videos.append(video)
            rgb_videos.sort()
            depth_videos.sort()
            for i in range(len(rgb_videos)):
                rgb_video_name = rgb_videos[i]
                depth_video_name = depth_videos[i]
                print(video_folder_path + '/' + rgb_video_name)
                main(
                    rgb_video_path = os.path.join(video_folder_path, rgb_video_name),
                    depth_video_path = os.path.join(video_folder_path, depth_video_name),
                    save_path = 'dataset/additional_dataset/other_people/hyemin/hyemin_npy_train/' + str(index) + '.' + action
                    )
    elif inference_mode:
        if not use_video:
            main()
        elif use_realsense:
            main()
        elif inference_mode_in_the_wild:
            base_path = os.getcwd()
            long_video_path = os.path.join(base_path, 'dataset', 'test_in_the_wild')
            rgb_videos = []
            depth_videos = []
            for file in os.listdir(long_video_path):
                if 'depth' in file:
                    depth_videos.append(file)
                else:
                    rgb_videos.append(file)
            rgb_videos.sort()
            depth_videos.sort()
            for i in range(len(rgb_videos)):
                main(
                rgb_video_path=long_video_path+'/'+rgb_videos[i],
                depth_video_path=long_video_path+'/'+ depth_videos[i],
                save_path = 'dataset/In_the_wild_test_result/'
                    )

        else:
            base_path = os.getcwd()
            actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+', 'left', 'left_up', 'up',
            'right_up', 'right', 'right_down', 'down', 'left_down', 'zoom_in', 'zoom_out','standard']
            rgb_videos = []
            depth_videos = []
            # Video extraction from the test video folder
            for index, action in enumerate(actions):
                video_folder_path = os.path.join(base_path, 'dataset', 'test_video', str(index) + '.' + action)
                for video in os.listdir(video_folder_path):
                    if 'depth' in video:
                        depth_videos.append(os.path.join(video_folder_path, video))
                    else:
                        rgb_videos.append(os.path.join(video_folder_path, video))

            # Video sequence shuffle
            length_array = np.arange(len(rgb_videos)).tolist()
            random.shuffle(length_array)
            rgb_videos = [rgb_videos[i] for i in length_array]
            depth_videos = [depth_videos[i] for i in length_array]

            for i in range(len(rgb_videos)):
                main(
                rgb_video_path=rgb_videos[i],
                depth_video_path=depth_videos[i],
                save_path = 'dataset/test_npy/' + str(index) + '.' + action
                    )

    elif use_video:
        base_path = os.getcwd()
        video_folder = 'dataset/additional_dataset/test_in_the_wild'
        action = 'looking_0.avi'
        rgb_path = os.path.join(base_path, video_folder, action)
        depth_action = 'looking_0_depth.avi'
        depth_path =  os.path.join(base_path, video_folder, depth_action)
        main(rgb_video_path= rgb_path, depth_video_path=depth_path, save_path='dataset/additional_dataset/low_level_feature_extraction_Result/')

    else:
        main()

if __name__ == "__main__":
    main_function()