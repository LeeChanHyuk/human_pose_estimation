##############################################################
################## 2022.06.02.ChanHyukLee ####################
##############################################################
# Facial & Body landmark is from mediaPipe (Google)
# Head pose estimation module is from 1996scarlet (https://github.com/1996scarlet/Dense-Head-Pose-Estimation)
# Gaze estimation module is from david-wb (https://github.com/david-wb/gaze-estimation)

from lib2to3.pytree import BasePattern
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import librar
import math
import cv2
import time
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import re
import random
import os
from data.human import HumanInfo

from sqlalchemy import true
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Import modules from directory
from utils.draw_utils import draw_axis
import utils.visualization_tool as visualization_tool
from utils.inference_module import inference
import head_pose_estimation_module.service as service
from gaze_estimation_module.util.gaze import draw_gaze
from gaze_estimation_module.gaze_estimation import estimate_gaze_from_face_image
import head_pose_estimatior
import body_pose_estimatior
from utils import preprocessing

# Mediapipe visualization objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Mode (If you want to use multiple functions, then make that you use true)
mode = 3
precise_value_visualization = True

# Landmark / action names
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


# This func will be developed with precise calibration code
def calibration(human_info, real_sense_calibration = True):
    center_eyes = human_info.center_eyes[-1].copy()
    calib_parameter = [0.9245, -0.004, 0.0584, -0.0242, 0.9475, -0.0083, 0.0208, 0.1013, 0.8956, -32.2596, 121.3725, 26.666, 0.008]
    # y = 240 - y
    # x = x - 320
    # for D435
    camera_horizontal_angle = 87 # RGB = 60
    camera_vertical_angle = 58 # RGB = 42

    i_width = 640
    i_height = 480
    
    # before calibration
    eye_x = center_eyes[0] - (i_width/2)
    eye_y = (i_height/2) - center_eyes[1]
    eye_z = center_eyes[2]

    detected_x_angle = (camera_horizontal_angle / 2) * (eye_x / (i_width/2))
    detected_y_angle = (camera_vertical_angle / 2) * (eye_y / (i_height/2))

    new_x = eye_z * math.sin(math.radians(detected_x_angle))
    new_y = eye_z * math.sin(math.radians(detected_y_angle))
    new_z = eye_z

    new_x, new_y, new_z = new_x * -1.0, new_y * 1.0, new_z * 1.0
    new_x = calib_parameter[0] * new_x + calib_parameter[3] * new_y + calib_parameter[6] * new_z + (calib_parameter[9])
    new_y = calib_parameter[1] * new_x + calib_parameter[4] * new_y + calib_parameter[7] * new_z + (calib_parameter[10])
    new_z = calib_parameter[2] * new_x + calib_parameter[5] * new_y + calib_parameter[8] * new_z + (calib_parameter[11])

    human_info.calib_center_eyes = [new_x, new_y, new_z]

    # Old calib
    #new_x = eye_z * math.sin(math.radians(detected_x_angle)) * 7 / 9
    #new_y = eye_z * math.sin(math.radians(detected_y_angle))
    #y_offset = eye_z * math.sin(math.radians(camera_vertical_angle/2))

def main_user_drawing(frame, human_infos, main_user_index):
    height, width = frame.shape[:2]
    #main_user_index = random.randint(0, len(face_box_per_man)-1) # For random main user visualization
    for index, human_info in enumerate(human_infos):
        if index == main_user_index:
            cv2.line(frame, (int(width/2), height-1), (int(human_info.center_eyes[-1][0]), int(human_info.center_eyes[-1][1])), (0, 255, 0), 3)
            cv2.rectangle(frame, (int(human_info.face_box[0][0]), int(human_info.face_box[0][1])), (int(human_info.face_box[0][2]), int(human_info.face_box[0][3])), (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'M', (int((human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(human_info.face_box[0][1])), 1, 2, (0, 255, 0), 2)
        else:
            cv2.line(frame, (int(width/2), height-1), (int(human_info.center_eyes[-1][0]), int(human_info.center_eyes[-1][1])), (255, 0, 0), 1)
            cv2.rectangle(frame, (int(human_info.face_box[0][0]), int(human_info.face_box[0][1])), (int(human_info.face_box[0][2]), int(human_info.face_box[0][3])), (255,0,0), 2, cv2.LINE_AA)
    return frame

def main_user_classification(frame, human_infos):
    man_score = [4000] * len(human_infos)
    for index, human_info in enumerate(human_infos):
        man_score[index] -= human_info.center_eyes[-1][2]
        if abs(human_info.head_poses[-1][0]) > 30: # if yaw value of the man is over than 30 or -30, then we think that the man is not looking the display.
            man_score[index] = 0
    max_val = max(man_score)
    main_user_index = man_score.index(max_val)
    draw_frame = main_user_drawing(frame.copy(), human_infos, main_user_index)
    return main_user_index, draw_frame
    
def realsense_initialization():
    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline = rs.pipeline()
    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline, align

def video_loader_initialization(path):
    files = os.listdir(path)
    rgb_videos = []
    depth_videos = []
    for file in files:
        if 'rgb' in file:
            rgb_videos.append(file)
        elif 'depth' in file:
            depth_videos.append(file)
    rgb_videos.sort()
    depth_videos.sort()
    rgb_caps = []
    depth_caps = []
    for rgb_video, depth_video in zip(rgb_videos, depth_videos):
        rgb_cap = cv2.VideoCapture(os.path.join(path, rgb_video))
        depth_cap = cv2.VideoCapture(os.path.join(path, depth_video))
        rgb_caps.append(rgb_cap)
        depth_caps.append(depth_cap)
    return rgb_caps, depth_caps, len(rgb_caps)

def load_mode(base_path):
    communication_file = open(os.path.join(base_path, 'communication.txt'), 'r')
    mode = communication_file.readline().strip()
    communication_file.close()
    return mode

def get_input(pipeline=None, align=None, rgb_cap=None, depth_cap=None, video_path=None):
    # Get input
    if not video_path:
        frames = pipeline.wait_for_frames()
        align_frames = align.process(frames)
        frame = align_frames.get_color_frame()
        depth = align_frames.get_depth_frame()
        depth = np.array(depth.get_data())
        frame = np.array(frame.get_data())
    else:
        ret, frame = rgb_cap.read()
        ret, depth = depth_cap.read()
    return frame, depth

def flag_initialization(human_info):
    human_info.face_detection_flag = False
    human_info.head_pose_estimation_flag = False
    human_info.body_pose_estimation_flag = False
    human_info.gaze_estimation_flag = False

def face_detection(frame, depth, face_mesh, human_infos = None):
    height, width = frame.shape[:2]
    bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(bgr_image)
    if face_results.multi_face_landmarks:
        if not human_infos:
            human_infos = []
        for index, face_landmarks in enumerate(face_results.multi_face_landmarks):
            if index >= len(human_infos):
                human_info = HumanInfo()
            else:
                human_info = human_infos[index]
                flag_initialization(human_info)
            face_boxes, left_eye_box, right_eye_box = head_pose_estimatior.box_extraction(
                face_landmarks=face_landmarks,
                width = width,
                height = height)
            face_box = np.array(face_boxes)
            left_eye_box = np.array(left_eye_box)
            right_eye_box = np.array(right_eye_box)
            human_info.face_box = face_box # face box is not used for action recognition. Thus, face_box is not list.
            human_info.left_eye_box = left_eye_box
            human_info.right_eye_box = right_eye_box

            center_eyes_x = (left_eye_box[0][0] + left_eye_box[0][2]) / 2
            center_eyes_y = (left_eye_box[0][1] + left_eye_box[0][3]) / 2
            center_eyes_z = depth[int(center_eyes_y), int(center_eyes_x)]
            human_info._put_data([center_eyes_x, center_eyes_y, center_eyes_z], 'center_eyes')
            if index >= len(human_infos):
                human_infos.append(human_info)
    if face_results.multi_face_landmarks:
        return human_infos, len(face_results.multi_face_landmarks)
    else:
        return human_infos, 0

def head_pose_estimation(frame, human_infos, fa, handler):
    feed = frame.copy()
    # Estimate head pose
    if head_pose_estimation:
        for index, human_info in enumerate(human_infos):
            face_box  = human_info.face_box
            for results in fa.get_landmarks(feed, face_box):
                pitch, yaw, roll = handler(frame, results, color=(125, 125, 125))
                human_info._put_data([yaw, pitch, roll], 'head_poses')
    return human_infos

def gaze_estimation(frame_copy, frame, human_info, visualization):
    frame, eyes = estimate_gaze_from_face_image(frame_copy, frame, human_info, visualization)
    left_eye, right_eye = eyes
    left_gaze = left_eye.gaze.copy()
    left_gaze[1] = -left_gaze[1]
    right_gaze = right_eye.gaze.copy()
    gazes = [left_gaze, right_gaze]

    human_info._put_data([gazes[0][0], gazes[0][1], gazes[1][0], gazes[1][1]], 'eye_poses')
    human_info.left_eye_landmark = left_eye
    human_info.right_eye_landmark = right_eye
    human_info.left_eye_gaze = left_gaze
    human_info.right_eye_gaze = right_gaze
    return frame

def body_pose_estimation(pose, frame, draw_frame, depth, human_info):
    height, width = frame.shape[:2]
    cv2.imshow('test', frame)
    cv2.waitKey(1)
    results = pose.process(frame)
    if results.pose_landmarks:
        body_landmarks= results.pose_landmarks
        body_landmarks = np.array([[lmk.x * width, lmk.y * height, lmk.z * width]
            for lmk in body_landmarks.landmark], dtype=np.float32)

        left_shoulder, right_shoulder, center_stomach, center_mouth, left_x_offset, left_y_offset, right_x_offset, right_y_offset, center_eye3 = body_pose_estimatior.body_keypoint_extractor(body_landmarks, landmark_names, depth, width, height)
        frame = visualization_tool.draw_body_keypoints(draw_frame, [left_shoulder, right_shoulder, center_stomach, center_mouth, center_eye3])
        upper_body_yaw, upper_body_pitch, upper_body_roll = body_pose_estimatior.upside_body_pose_calculator(left_shoulder, right_shoulder, center_stomach)
        upper_body_yaw = upper_body_yaw * 180 / math.pi
        upper_body_pitch = upper_body_pitch * 180 / math.pi
        upper_body_roll = upper_body_roll * 180 / math.pi

        human_info._put_data(center_stomach, 'center_stomachs')
        human_info._put_data(center_mouth, 'center_mouths')
        human_info._put_data(left_shoulder, 'left_shoulders')
        human_info._put_data(right_shoulder, 'right_shoulders')
        human_info._put_data([upper_body_yaw, upper_body_pitch, upper_body_roll], 'body_poses')
    return draw_frame

def visualization(frame, depth, human_info):
    height, width = frame.shape[:2]
    # apply colormap to depthmap
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

    # Visualize head pose
    if human_info.head_pose_estimation_flag:
        frame = draw_axis(frame, human_info.head_poses[-1][0], human_info.head_poses[-1][1], human_info.head_poses[-1][2], 
                          [int((human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(human_info.face_box[0][1] - 30)])

    # Visualize body pose
    if human_info.body_pose_estimation_flag:
        frame = draw_axis(frame, human_info.body_poses[-1][0], human_info.body_poses[-1][1], human_info.body_poses[-1][2], 
                          [int((human_info.left_shoulders[-1][0] + human_info.right_shoulders[-1][0])/2), int(human_info.left_shoulders[-1][1])], 
                          color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))

    # Visualize eye pose
    if human_info.gaze_estimation_flag:
        for i, ep in enumerate([human_info.left_eye_landmark, human_info.right_eye_landmark]):
            for (x, y) in ep.landmarks[16:33]:
                color = (0, 255, 0)
                if ep.eye_sample.is_left:
                    color = (255, 0, 0)
                cv2.circle(frame,(int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
            gaze = [human_info.left_eye_gaze, human_info.right_eye_gaze][i]
            length = 60.0
            draw_gaze(frame, ep.landmarks[-2], gaze, length=length, thickness=2)

    # Visualize the values of each poses
    if precise_value_visualization:
        width = int(width * 1.5)
        zero_array = np.zeros((height, width, 3), dtype=np.uint8)
        if human_info.body_pose_estimation_flag:
            center_shoulder = (human_info.left_shoulders[-1] + human_info.right_shoulders[-1]) / 2
            zero_array= visualization_tool.draw_body_information(zero_array, width, height, round(center_shoulder[0], 2), round(center_shoulder[1], 2), round(center_shoulder[2], 2), 
                                                                 round(human_info.body_poses[-1][0], 2), round(human_info.body_poses[-1][1], 2), round(human_info.body_poses[-1][2], 2))
        if human_info.head_pose_estimation_flag:
            zero_array = visualization_tool.draw_face_information(zero_array, width, height, round(human_info.center_eyes[-1][0], 2), round(human_info.center_eyes[-1][1]),
                                                                  round(human_info.center_eyes[-1][2]), round(human_info.head_poses[-1][0], 2), round(human_info.head_poses[-1][1], 2),
                                                                  round(human_info.head_poses[-1][2], 2))

        if human_info.gaze_estimation_flag:
            zero_array = visualization_tool.draw_gaze_information(zero_array,width, height, round(human_info.center_eyes[-1][0], 2), round(human_info.center_eyes[-1][1], 2),
                                                                  round(human_info.center_eyes[-1][2], 2), [human_info.left_eye_gaze, human_info.right_eye_gaze])
        stacked_frame = np.concatenate([frame, zero_array], axis=1)
        return stacked_frame
    else:
        return None

def action_recognition(frame, draw_frame, human_info, fps):
    fps = int(fps)
    if human_info.body_pose_estimation_flag and human_info.head_pose_estimation_flag and fps>0:
        center_eyes = np.array(human_info.center_eyes[-2*fps:])
        center_mouths = np.array(human_info.center_mouths[-2*fps:])
        left_shoulders = np.array(human_info.left_shoulders[-2*fps:])
        right_shoulders = np.array(human_info.right_shoulders[-2*fps:])
        center_stomachs = np.array(human_info.center_stomachs[-2*fps:])
        head_poses = np.array(human_info.head_poses[-2*fps:])
        network_input = np.array([center_eyes, center_mouths, left_shoulders, right_shoulders, center_stomachs, head_poses])

        # 총 25개의 features
        network_input = preprocessing.data_preprocessing(network_input, fps)
        network_input = np.expand_dims(np.array(network_input),axis=0)
        human_state = inference(network_input)
        human_info.human_state = human_state
        draw_frame = cv2.flip(draw_frame, 1)
        cv2.putText(draw_frame, human_state, (0, 50), 1, 3, (0, 0, 255), 3)
        return draw_frame
    else:
        print("The body and head pose are not estimated normally. Please check the state of the human.")
        return frame

def networking(human_info, mode, base_path):
    communication_write = open(os.path.join(base_path, 'communication.txt'), 'r+')
    communication_write.write(mode)
    communication_write.write(str(round(human_info.center_eyes[-1][0])).zfill(3) + ' ' + str(round(human_info.center_eyes[-1][1]+20)).zfill(3)
                              + ' ' + str(round(human_info.center_eyes[-1][2])).zfill(3) + '\n')
    communication_write.write(str(round(human_info.head_poses[-1][1])).zfill(3) + ' ' + str(round(human_info.head_poses[-1][0])).zfill(3)
                              + ' ' + str(round(human_info.head_poses[-1][2])).zfill(3) + '\n')
    communication_write.write(human_info.human_state+'\n')
    communication_write.close()


def main(video_folder_path=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    fps = 20
    iteration = 0
    human_infos = None

    # Initialize face detection module
    fa = service.DepthFacialLandmarks(os.path.join(base_path, "head_pose_estimation_module/weights/sparse_face.tflite"))
    print('Face detection module is initialized')

    # Initialize head pose estimation module
    handler = getattr(service, 'pose')
    print('Head pose estimation module is initialized')

    if not video_folder_path:
        pipeline, align = realsense_initialization()
    # Video
    else:
        rgb_caps, depth_caps, total_video_num = video_loader_initialization(video_folder_path)
        current_video_index = 0
        rgb_cap, depth_cap = rgb_caps[current_video_index], depth_caps[current_video_index]
    # Define pose estimation & face detection thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            min_detection_confidence=0.5) as face_mesh:
            print("Initialization step is done. Please turn on the super multiview renderer")

            while True:
                # Load mode (0: No tracking / 1: Eye tracking / 2: Eye tracking + Head pose estimation / 3: Eye tracking + Head pose estimation + Action recongition)
                mode = load_mode(base_path=base_path) # mode
                if mode == 0:
                    break

                # Get input
                start_time = time.time()
                if not video_folder_path:
                    frame, depth = get_input(pipeline=pipeline, align=align, video_path=video_folder_path)
                else: # Load next video automatically.
                    (rgb_ret, frame), (depth_ret, depth) = rgb_cap.read(), depth_cap.read()
                    if depth_ret:
                        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
                    else:
                        if current_video_index + 1 < total_video_num:
                            current_video_index += 1
                            rgb_cap, depth_cap = rgb_caps[current_video_index], depth_caps[current_video_index]
                            continue
                        else:
                            break

                if not frame.any() or not depth.any():
                    continue
                if depth.shape != frame.shape:
                    frame = cv2.resize(frame, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

                # Get frame information
                height, width = frame.shape[:2]
                
                # Input visualization
                frame_copy = frame.copy()
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame_copy)
                cv2.waitKey(1)

                # Media pipe face detection
                human_infos, face_num = face_detection(frame, depth, face_mesh, human_infos)

                if face_num:
                    # Head pose estimation
                    human_infos = head_pose_estimation(frame, human_infos, fa, handler)
                        
                    # Main user classification
                    main_user_index, draw_frame = main_user_classification(frame, human_infos)
                    human_infos = [human_infos[main_user_index]]

                    # Gaze estimation
                    #frame = gaze_estimation(frame_copy, frame, human_infos[main_user_index], visualization)

                    # Body pose estimation
                    draw_frame = body_pose_estimation(pose, frame, draw_frame, depth, human_infos[0])

                    # Visualization
                    #stacked_frame = visualization(frame, depth, human_infos[main_user_index])
                    
                    # Action recognition
                    draw_frame = action_recognition(frame, draw_frame, human_infos[0], fps)

                    # Calibration
                    calibration(human_infos[0])

                    # Networking with renderer
                    networking(human_infos[0], mode, base_path)

                cv2.namedWindow('MediaPipe Pose1', cv2.WINDOW_NORMAL)
                cv2.imshow('MediaPipe Pose1', draw_frame)

                fps = 1 / (time.time() - start_time)
                #print(fps)
                pressed_key = cv2.waitKey(1)
                if pressed_key == 27:
                    break

def main_function():
    main(video_folder_path='C:/Users/user/Desktop/test')
    #main()

if __name__ == "__main__":
    main_function()