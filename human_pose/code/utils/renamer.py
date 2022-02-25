import os

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_train_npy/4.standard'

for file in os.listdir(train_file_path):
    os.rename(os.path.join(train_file_path, file),
    os.path.join(train_file_path,'4standard_'+file))