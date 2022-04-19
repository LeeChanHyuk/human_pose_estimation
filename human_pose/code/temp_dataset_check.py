import numpy as np
import os

path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/train_npy'

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        np_file = np.load(os.path.join(path, folder, file))
        depth = np_file[0][0][2]
        if depth>200:
            print(depth)