import os

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/train_npy/17.looking'

for file in os.listdir(train_file_path):
    os.rename(os.path.join(train_file_path, file),
    os.path.join(train_file_path,'look_'+file))