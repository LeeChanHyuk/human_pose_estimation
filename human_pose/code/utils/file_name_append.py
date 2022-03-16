import os
import numpy as np
import re

path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_dataset/other_people/hyemin'
for big_folder in os.listdir(path):
    for folder in os.listdir(os.path.join(path, big_folder)):
        for file in os.listdir(os.path.join(path, big_folder, folder)):
            os.rename(os.path.join(path, big_folder, folder, file),
            os.path.join(path, big_folder, folder, 'h'+file))