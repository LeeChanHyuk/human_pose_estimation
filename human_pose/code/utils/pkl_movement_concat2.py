import pickle
import os

path = '/home/ddl/git/human_pose_estimation/human_pose/mmaction2/mmaction2/human_pose/train_data.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)
    print(data['04798'])
    print(1)
print(1)