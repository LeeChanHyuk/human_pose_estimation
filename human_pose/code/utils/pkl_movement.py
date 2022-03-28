import os
import numpy as np
import re

train_path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/train_video'
test_path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/test_video'
train_pkl_path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/train_pkl'
test_pkl_path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/test_pkl'

for file in os.listdir(train_path):
    if 'pkl' in file:
        os.rename(os.path.join(train_path, file),
        os.path.join(train_pkl_path, file))

for file in os.listdir(test_path):
    if 'pkl' in file:
        os.rename(os.path.join(test_path, file),
        os.path.join(test_pkl_path, file))