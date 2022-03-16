import os
import numpy as np
import re

path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/test_video/test_rgb_video'
save_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/test_video/test_depth_video'

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        if 'depth' in file:
            os.rename(os.path.join(path, folder, file),
            os.path.join(save_path, folder, file))
