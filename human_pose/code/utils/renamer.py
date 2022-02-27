import os

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_dataset/temp_npy'

for folder in os.listdir(train_file_path):
    for file in os.listdir(os.path.join(train_file_path, folder)):
        os.rename(os.path.join(train_file_path, folder, file),
        os.path.join(train_file_path,folder, folder[2:]+file))