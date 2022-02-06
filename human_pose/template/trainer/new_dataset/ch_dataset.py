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
        return self.length

    def get_data(self):
        data_file_names = os.listdir(self.data_path)
        temp_data= []
        data = []
        labels = []
        neutral = 0
        yaw = 0
        pitch = 0
        sequence_length = 30
        for file in data_file_names:
            f = open(os.path.join(self.data_path, file), 'r')
            while 1:
                line = f.readline()
                if not line: break
                yaw, pitch, roll, label = line.strip().split(' ')
                yaw, pitch, roll = float(yaw), float(pitch), float(roll)
                if label == 'N' and (yaw > 0):
                    motion = yaw - int(yaw / 3)
                    neutral = yaw - motion
                    for i in range(neutral):
                        labels.append(0)
                    for i in range(motion):
                        labels.append(1)
                    yaw = 0
                elif label == 'N' and (pitch>0):
                    motion = pitch - int(pitch / 3)
                    neutral = pitch - motion
                    for i in range(neutral):
                        labels.append(0)
                    for i in range(motion):
                        labels.append(2)
                    pitch = 0
                elif label == 'Y':
                    yaw += 1
                elif label == 'P':
                    pitch += 1
                else:
                    labels.append(0)
                temp_data.append([yaw, pitch, roll])
            for i in range(len(temp_data) - (sequence_length-1)):
                data.append(temp_data[i:i+sequence_length])
            labels = labels[(sequence_length-1):]
        return np.array(data), np.array(label)
                

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
