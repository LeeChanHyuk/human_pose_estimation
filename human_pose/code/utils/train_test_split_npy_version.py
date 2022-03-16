import os
import random
import re

"""
Note that the train_video, train_npy, test_video, test_npy folder must be in dataset folder
and the '***_video' folder must be filled by several videos
"""

base_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_dataset/other_people/hyemin'
train_npy_path = os.path.join(base_path, 'train_npy')
test_npy_path = os.path.join(base_path, 'test_npy')
paths = [train_npy_path, test_npy_path]

for folder in os.listdir(train_npy_path):
    rgb_npys=[]
    rgb_file_names = []
    depth_file_names = []
    for file in os.listdir(os.path.join(base_path, train_npy_path, folder)):
        rgb_npys.append(file)
    for i in range(len(rgb_npys)):
        rgb_numbers = re.sub(r'[^0-9]', '', rgb_npys[i]).zfill(3)
        rgb_string = ''.join([i for i in rgb_npys[i] if not i.isdigit()])
        new_rgb_file_name = str(rgb_numbers) + '_' + rgb_string
        os.rename(os.path.join(train_npy_path, folder, rgb_npys[i]), os.path.join(train_npy_path, folder, new_rgb_file_name))
        os.rename(os.path.join(test_npy_path, folder, rgb_npys[i]), os.path.join(test_npy_path, folder, new_rgb_file_name))
        rgb_file_names.append(new_rgb_file_name)
    rgb_file_names.sort()
    depth_file_names.sort()
    alist=[]                            
    for i in range(int(0.2 * len(rgb_npys))):
        a = random.randint(0, len(rgb_npys)-1)
        while a in alist :             
            a = random.randint(0,len(rgb_npys)-1)
        alist.append(a)
    for i in range(len(rgb_npys)):
        train_rgb_file_name = os.path.join(train_npy_path, folder, rgb_file_names[i])
        test_rgb_file_name = os.path.join(test_npy_path, folder, rgb_file_names[i])
        if i in alist:
            os.remove(train_rgb_file_name)
        else:
            os.remove(test_rgb_file_name)
