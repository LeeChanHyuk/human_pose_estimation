import os
import numpy as np

file_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/train_npy/8.left_up/000.npy'

file = np.load(file_path)
print(file)