# init에서는 dataset을 create하는 함수를 만들어야지.
import logging
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
        data_file_names = os.listdir(self.data_path)
        temp_data= []
        data = []
        labels = []
        neutral = 0
        yaw_count = 0
        pitch_count = 0
        sequence_length = 30
        line_count = 0
        for file in data_file_names:
            f = open(os.path.join(self.data_path, file), 'r')
            while 1:
                line = f.readline()
                if not line: break
                line_count += 1
                yaw, pitch, roll, label = line.strip().split(' ')
                yaw, pitch, roll = float(yaw), float(pitch), float(roll)
                if label == 'N' and (yaw_count > 0):
                    motion = int(yaw_count - int(yaw_count * 2 / 3))
                    neutral = int(yaw_count - motion)
                    for i in range(neutral):
                        labels.append(0)
                    for i in range(motion):
                        labels.append(1)
                    yaw_count = 0
                    labels.append(0)
                elif label == 'N' and (pitch_count>0):
                    motion = int(pitch_count - int(pitch_count * 2 / 3))
                    neutral = int(pitch_count - motion)
                    for i in range(neutral):
                        labels.append(0)
                    for i in range(motion):
                        labels.append(2)
                    pitch_count = 0
                    labels.append(0)
                elif label == 'Y':
                    yaw_count += 1
                elif label == 'P':
                    pitch_count += 1
                else:
                    labels.append(0)
                temp_data.append([yaw, pitch, roll])
        for i in range(len(temp_data) - (sequence_length-1)):
            data.append(temp_data[i:i+sequence_length]) # 1462 / 1433, 3851 + 1462 = 5313 / 5255
        labels = labels[(sequence_length-1):]
        return np.array(data), np.array(labels)
                

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
