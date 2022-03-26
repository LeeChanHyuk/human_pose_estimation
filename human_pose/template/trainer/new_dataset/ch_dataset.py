# init에서는 dataset을 create하는 함수를 만들어야지.
from fileinput import filename
import logging
from tkinter import N
from matplotlib.pyplot import axis
import torch
import torchvision
import torch.utils.data
import hydra
import glob
import re
import numpy as np
import os
import pandas as pd
import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from PIL import Image
import collections

class ch_dataset(torch.utils.data.Dataset):
    def __init__(self, conf, mode) -> None:
        super().__init__() # 아무것도 안들어있으면 자기 자신을 호출 (왜?)
        self.conf = conf[mode]
        self.mode = mode
        self.data_path = self.conf['dataset_path']
        self.data, self.labels = self.get_data()
        self.train_file_list = []

    def __len__(self):
        return len(self.data)

    def get_data(self):
        data = []
        labels = []
        normalize_length = 3
        for index, action in enumerate(os.listdir(self.data_path)):
            index = int(re.sub(r'[^0-9]', '', action))
            for pose in os.listdir(os.path.join(self.data_path, action)):
                """Normalize standardrizaton
                transition_x : image_width
                transition_y : image_height
                head, body poses : 90
                gaze_poses : 1
                """
                poses_from_one_video = np.load(os.path.join(self.data_path, action, pose))
                center_eyes = poses_from_one_video[0, :, :3]
                center_mouths = poses_from_one_video[1, :, :3]
                left_shoulders = poses_from_one_video[2, :, :3]
                right_shoulders = poses_from_one_video[3, :, :3]
                center_stomachs = poses_from_one_video[4, :, :3]

                # transition normalization
                for i in range(center_eyes.shape[-1]):
                    if i == 0:
                        div_num = 640
                    elif i == 1:
                        div_num = 480
                    else:
                        div_num = 256
                    center_eyes[:,i] /= div_num
                    center_mouths[:,i] /= div_num
                    left_shoulders[:,i] /= div_num
                    right_shoulders[:,i] /= div_num
                    center_stomachs[:,i] /= div_num
                    
                head_poses = poses_from_one_video[5, :, :3] / 90
                body_poses = poses_from_one_video[6, :, :3] / 90
                gaze_poses = poses_from_one_video[7]
                all_poses = np.concatenate([center_eyes, center_mouths, left_shoulders, right_shoulders, center_stomachs, head_poses, body_poses, gaze_poses], axis=1)
                normalized_poses = []
                for i in range(all_poses.shape[-1]):
                    i_all = all_poses[:,i]
                    i_1 = i_all[0:len(i_all):3]
                    i_2 = i_all[1:len(i_all):3]
                    i_3 = i_all[2:len(i_all):3]
                    while len(i_1) < all_poses.shape[0] / 3:
                        i_1 = np.concatenate([i_1, np.expand_dims(np.array(i_1[-1]), axis=0)], axis=0)
                    while len(i_2) < all_poses.shape[0] / 3:
                        i_2 = np.concatenate([i_2, np.expand_dims(np.array(i_2[-1]), axis=0)], axis=0)
                    while len(i_3) < all_poses.shape[0] / 3:
                        i_3 = np.concatenate([i_3, np.expand_dims(np.array(i_3[-1]), axis=0)], axis=0)
                    i_all = (i_2 + i_2 + i_3) / 3
                    normalized_poses.append(i_all)
                normalized_poses = np.array(normalized_poses).transpose()
                normalized_poses = self.data_normalization(normalized_poses)
                data.append(normalized_poses)
                labels.append(index)
                if self.conf['reverse_augmentation']:
                    if index >=0 and index <= 6:
                        data.append(normalized_poses[::-1])
                        labels.append(index)
                    #elif index != 15 and index != 16:
                    #    data.append(normalized_poses[::-1])
                    #    labels.append(index)

        return np.array(data), np.array(labels)

    def minmax_normalization(self, data : np.array) -> np.array:
        for i in range(data.shape[-1]):
            data[:,i] = (data[:,i] - min(data[:,i])) / (max(data[:,i] - min(data[:,i])))
        return data

    def data_normalization(self, data : np.array) -> np.array:
        for i in range(data.shape[-1]):
            data[:, i] -= data[0, i]
        return data

    def preprocessing_for_embedding(self, data: np.array, quantization_number: int, min=None, max=None) -> np.array:
        data = data // quantization_number
        data = data * quantization_number
        if min is not None:
            temp_matrix = np.ones_like(data) * min
            data = np.where(data < min, min, data)
        if max is not None:
            temp_matrix = np.ones_like(data) * max
            data = np.where(data > max, max, data)
        return data



                

    # data augmentation is conducted in here because of probability of augmentation method
    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.data[index]
            label = self.labels[index]
            return data, label
            
        else: # for test dataset
            data = self.data[index]
            label = self.labels[index]
            return data, label
