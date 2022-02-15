# init에서는 dataset을 create하는 함수를 만들어야지.
import logging
from matplotlib.pyplot import axis
import torch
import torchvision
import torch.utils.data
import hydra
import glob
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

    def __len__(self):
        return len(self.data)

    def get_data(self):
        data = []
        labels = []
        normalize_length = 3
        for index, action in enumerate(os.listdir(self.data_path)):
            for pose in os.listdir(os.path.join(self.data_path, action)):
                poses_from_one_video = np.load(os.path.join(self.data_path, action, pose))
                normalized_pose = []
                for i in range(poses_from_one_video.shape[-1]):
                    i_all = poses_from_one_video[:,i]
                    i_1 = i_all[0:len(i_all):3]
                    i_2 = i_all[1:len(i_all):3]
                    i_3 = i_all[2:len(i_all):3]
                    while len(i_1) < 20:
                        i_1 = np.concatenate([i_1, np.expand_dims(np.array(i_1[-1]), axis=0)], axis=0)
                    while len(i_2) < 20:
                        i_2 = np.concatenate([i_2, np.expand_dims(np.array(i_2[-1]), axis=0)], axis=0)
                    while len(i_3) < 20:
                        i_3 = np.concatenate([i_3, np.expand_dims(np.array(i_3[-1]), axis=0)], axis=0)
                    i_all = (i_2 + i_2 + i_3) / 3
                    normalized_pose.append(i_all)
                normalized_pose = np.array(normalized_pose).transpose()
                normalized_pose = self.data_normalization(normalized_pose)
                data.append(normalized_pose)
                labels.append(index)

        return np.array(data), np.array(labels)

    def data_normalization(self, data : np.array):
        for i in range(data.shape[-1]):
            data[:,i] = (data[:,i] - min(data[:,i])) / (max(data[:,i] - min(data[:,i])))
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
