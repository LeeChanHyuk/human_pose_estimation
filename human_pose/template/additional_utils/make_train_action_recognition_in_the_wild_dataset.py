from asyncio.proactor_events import BaseProactorEventLoop
from lib2to3.pytree import BasePattern
import os
import numpy as np
import cv2
import re

base_path = os.getcwd()
video_path = 'dataset/additional_dataset/other_people/hyemin_video'
save_path = 'dataset/additional_dataset/other_people/hyemin_result'
rgb_video_paths = []
depth_video_paths = []
actions = [ '0.nolooking', '1.yaw-', '2.yaw+', '3.pitch-', '4.pitch+', '5.roll-', '6.roll+', '7.left', '8.left_up', '9.up',
'10.right_up', '11.right', '12.right_down', '13.down', '14.left_down', '15.zoom_in', '16.zoom_out', '17.standard']

# Video append to two lists
for video in os.listdir(os.path.join(base_path, video_path)):
    if 'depth' in video:
        depth_video_paths.append(video)
    else:
        rgb_video_paths.append(video)

# Video list sort
depth_video_paths.sort()
rgb_video_paths.sort()

# buffers shape = (60, h, w, c)
rgb_buffers = [] 
depth_buffers = []
rgb_videos_name = []
depth_videos_name = []
rgb_video_list = []
depth_video_list = []
w = 640
h = 480
fps = 30 # 카메라에 따라 값이 정상적, 비정상적

def save_the_video(action, rgb_frame_list, depth_frame_list):
    num = int(len(os.listdir(os.path.join(base_path, save_path, action))) / 2)
    video_num = str(num).zfill(3)
    rgb_path = os.path.join(base_path, save_path, action, video_num+'_rgb.avi')
    depth_path = os.path.join(base_path, save_path, action, video_num+'_depth.avi')

    # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # 1프레임과 다음 프레임 사이의 간격 설정
    delay = round(1000/fps)

    # 웹캠으로 찰영한 영상을 저장하기
    # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    rgb_out = cv2.VideoWriter(rgb_path, fourcc, fps, (w, h))
    depth_out = cv2.VideoWriter(depth_path, fourcc, fps, (w, h))

    for frame in rgb_frame_list:
        rgb_out.write(frame)

    for frame in depth_frame_list:
        depth_out.write(frame)

    rgb_out.release()
    depth_out.release()

        
for i in range(len(rgb_video_paths)):
    rgb_video_path = os.path.join(base_path, video_path, rgb_video_paths[i])
    depth_video_path = os.path.join(base_path, video_path, depth_video_paths[i])
    rgb_video = cv2.VideoCapture(rgb_video_path)
    depth_video = cv2.VideoCapture(depth_video_path)
    frame_count = 0
    while 1:
        ret, rgb_frame = rgb_video.read()
        ret, depth_frame = depth_video.read()
        frame_count += 1
        print(frame_count)
        if rgb_frame is None:
            break
        rgb_buffers.append(rgb_frame)
        depth_buffers.append(depth_frame)
        if len(rgb_buffers) == 60:
            while 1:
                for j in range(len(rgb_buffers)):
                    cv2.imshow('rgb_frame', rgb_buffers[j])
                    cv2.waitKey(20)
                key_board_input = cv2.waitKey(0)
                state = -1
                if key_board_input == ord('u'):
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    break
                elif key_board_input == ord('n'):
                    state = 0
                elif key_board_input == ord('r'):
                    state = 1
                elif key_board_input == ord('t'):
                    state = 2
                elif key_board_input == ord('f'):
                    state = 3
                elif key_board_input == ord('g'):
                    state = 4
                elif key_board_input == ord('v'):
                    state = 5
                elif key_board_input == ord('b'):
                    state = 6
                elif key_board_input == ord('a'):
                    state = 7
                elif key_board_input == ord('q'):
                    state = 8
                elif key_board_input == ord('w'):
                    state = 9
                elif key_board_input == ord('e'):
                    state = 10
                elif key_board_input == ord('d'):
                    state = 11
                elif key_board_input == ord('c'):
                    state = 12
                elif key_board_input == ord('x'):
                    state = 13
                elif key_board_input == ord('z'):
                    state = 14
                elif key_board_input == ord('s'):
                    state = 17
                elif key_board_input == ord('y'):
                    state = 15
                elif key_board_input == ord('h'):
                    state = 16
                if state > -1:
                    save_the_video(actions[state], rgb_buffers, depth_buffers)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    rgb_buffers.pop(0)
                    depth_buffers.pop(0)
                    break
