import os

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/train_video'

actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+', 'left_up',
'right_up',  'right_down', 'left_down', 'left', 'up', 'right', 'down', 'zoom_in', 'zoom_out', 'standard']
indexes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for index, file in enumerate(os.listdir(os.path.join(train_file_path))):
    for index, action in enumerate(actions):
        if action in file:
            file_name = str(indexes[index]).zfill(5) + '_' + 'A'+ str(index).zfill(3)+'.avi'
            indexes[index] += 1
            break
    os.rename(os.path.join(train_file_path, file),
    os.path.join(train_file_path, file_name))