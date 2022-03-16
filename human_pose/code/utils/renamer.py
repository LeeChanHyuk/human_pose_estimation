import os

train_file_path = '/home/ddl/git/human_pose_estimation/human_pose/dataset/additional_dataset/other_people/researcher_npy'

for folder in os.listdir(train_file_path):
    for index, file in enumerate(os.listdir(os.path.join(train_file_path, folder))):
        os.rename(os.path.join(train_file_path, folder, file),
        os.path.join(train_file_path, folder, 'img_' + file))