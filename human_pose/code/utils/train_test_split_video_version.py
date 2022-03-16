import os
import random
import re

"""
Note that the train_video, train_npy, test_video, test_npy folder must be in dataset folder
and the '***_video' folder must be filled by several videos
"""

base_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_dataset/other_people/hyemin'
train_video_path = os.path.join(base_path, 'train_video')
test_video_path = os.path.join(base_path, 'test_video')
train_npy_path = os.path.join(base_path, 'train_npy')
test_npy_path = os.path.join(base_path, 'test_npy')
paths = [train_video_path, test_video_path]

for folder in os.listdir(train_video_path):
    rgb_videos=[]
    depth_videos=[]
    rgb_file_names = []
    depth_file_names = []
    for file in os.listdir(os.path.join(base_path, train_video_path, folder)):
        if 'depth' in file:
            depth_videos.append(file)
        else:
            rgb_videos.append(file)
    for i in range(len(depth_videos)):
        rgb_numbers = re.sub(r'[^0-9]', '', rgb_videos[i]).zfill(3)
        depth_numbers = re.sub(r'[^0-9]', '', depth_videos[i]).zfill(3)
        rgb_string = ''.join([i for i in rgb_videos[i] if not i.isdigit()])
        depth_string = ''.join([i for i in depth_videos[i] if not i.isdigit()])
        new_rgb_file_name = str(rgb_numbers) + '_' + rgb_string
        new_depth_file_name = str(depth_numbers) + '_' + depth_string
        os.rename(os.path.join(train_video_path, folder, rgb_videos[i]), os.path.join(train_video_path, folder, new_rgb_file_name))
        os.rename(os.path.join(test_video_path, folder, rgb_videos[i]), os.path.join(test_video_path, folder, new_rgb_file_name))
        os.rename(os.path.join(train_video_path, folder, depth_videos[i]), os.path.join(train_video_path, folder, new_depth_file_name))
        os.rename(os.path.join(test_video_path, folder, depth_videos[i]), os.path.join(test_video_path, folder, new_depth_file_name))
        rgb_file_names.append(new_rgb_file_name)
        depth_file_names.append(new_depth_file_name)
    rgb_file_names.sort()
    depth_file_names.sort()
    assert len(depth_videos) == len(rgb_videos)
    alist=[]                            
    for i in range(int(0.2 * len(depth_videos))):
        a = random.randint(0, len(depth_videos)-1)
        while a in alist :             
            a = random.randint(0,len(depth_videos)-1)
        alist.append(a)
    for i in range(len(depth_videos)):
        train_rgb_file_name = os.path.join(train_video_path, folder, rgb_file_names[i])
        train_depth_file_name = os.path.join(train_video_path, folder, depth_file_names[i])
        test_rgb_file_name = os.path.join(test_video_path, folder, rgb_file_names[i])
        test_depth_file_name = os.path.join(test_video_path, folder, depth_file_names[i])
        if i in alist:
            os.remove(train_rgb_file_name)
            os.remove(train_depth_file_name)
        else:
            os.remove(test_rgb_file_name)
            os.remove(test_depth_file_name)
