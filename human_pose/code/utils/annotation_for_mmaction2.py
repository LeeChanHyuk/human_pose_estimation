import os
import cv2

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose'

"""for folder in os.listdir(train_file_path):
    for index, file in enumerate(os.listdir(os.path.join(train_file_path, folder))):
       os.rename(os.path.join(train_file_path, folder, file),
        os.path.join(train_file_path, 'all', file))"""
actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+',  'left_up', 
'right_up', 'right_down', 'left_down','left', 'right', 'up','down','zoom_in', 'zoom_out', 'standard']
f = open('/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/test_label_for_raw_frames.txt', 'w')
for file in os.listdir(os.path.join(train_file_path, 'test_video')):
    cap = cv2.VideoCapture(os.path.join(train_file_path, 'test_video', file))
    os.mkdir(os.path.join(train_file_path, 'frames', file[0:len(file)-4]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_name = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imwrite(os.path.join(train_file_path, 'frames', file[0:len(file)-4], str(frame_name)+'.jpg'), frame)
        frame_name += 1
    for action in actions:
        if action in file:
            index = actions.index(action)
            f.writelines(file[0:len(file)-4] + ' ' + str(frame_count) + ' '+ str(index)+'\n')
            break