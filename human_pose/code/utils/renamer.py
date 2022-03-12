import os

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_dataset/dataset/third_shot/npy'

for folder in os.listdir(train_file_path):
    for index, file in enumerate(os.listdir(os.path.join(train_file_path, folder))):
        os.rename(os.path.join(train_file_path, folder, file),
        os.path.join(train_file_path,folder, folder[3:]+str(index)+'.npy'))