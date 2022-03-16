# init에서는 dataset을 create하는 함수를 만들어야지.
# init에서 create가 들어가는 이유는 만들고자하는 dataset이 하나가 아니라 여러 개 일 때, 여기서 총체적으로 관리할 수 있기 때문이다.
# 즉, dataset을 만들 때 필요한 정보 관리 등이나 인자들을 여기서 만들어서 전달하자.

import logging

from cv2 import validateDisparity
from .ch_dataset import ch_dataset
import torch.utils.data
import torch
import torchvision
import torchvision.transforms as transforms
import mpose
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset
import numpy as np

def create(conf, local_rank, world_size, mode='train'):

    if conf[mode]['name'] == 'dataset':
        if mode == 'train' or mode == 'valid':
            dataset = ch_dataset(conf, mode)
            length = len(dataset)
            train_length = int(length * 0.8)
            valid_length = length - train_length
            train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
        else:
            dataset = ch_dataset(conf, mode)
            temp_dataset = dataset
                
    elif conf[mode]['name'] == 'mnist':
        if mode == 'train':
            temp_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
            train=True,
            transform=transforms .ToTensor(),
            download=True)
        else:
            temp_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True)
    elif conf[mode]['name'] == 'openpose':
        if mode == 'train':
            X_train, y_train, X_test, y_test = load_mpose('openpose', 1, verbose=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            test_size=0.1,
                                                            random_state=11331,
                                                            stratify=y_train)
            tensor_x = torch.Tensor(X_train) # transform to torch tensor
            tensor_y = torch.Tensor(y_train)
            temp_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        elif mode == 'valid':
            X_train, y_train, X_test, y_test = load_mpose('openpose', 1, verbose=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            test_size=0.1,
                                                            random_state=11331,
                                                            stratify=y_train)
            tensor_x = torch.Tensor(X_val) # transform to torch tensor
            tensor_y = torch.Tensor(y_val)
            temp_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        else:
            X_train, y_train, X_test, y_test = load_mpose('openpose', 1, verbose=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            test_size=0.1,
                                                            random_state=11331,
                                                            stratify=y_train)
            tensor_x = torch.Tensor(X_test) # transform to torch tensor
            tensor_y = torch.Tensor(y_test)
            temp_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    
    if conf[mode]['name'] == 'dataset':
        if mode == 'train' or mode == 'valid':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, 
                    num_replicas=world_size, 
                    rank=local_rank,
                    shuffle=(mode == 'train')
                )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=conf[mode]['batch_size'],
                shuffle=False,
                pin_memory=True,
                drop_last=conf[mode]['drop_last'],
                num_workers=0,
                sampler=train_sampler
            )

            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                    valid_dataset, 
                    num_replicas=world_size, 
                    rank=local_rank,
                    shuffle=(mode == 'train')
                )
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=conf[mode]['batch_size'],
                shuffle=False,
                pin_memory=True,
                drop_last=conf[mode]['drop_last'],
                num_workers=0,
                sampler=valid_sampler
            )
            return train_dataloader, train_sampler, valid_dataloader, valid_sampler

    sampler = torch.utils.data.distributed.DistributedSampler(
            temp_dataset, 
            num_replicas=world_size, 
            rank=local_rank,
            shuffle=(mode == 'train')
        )
    dataloader = torch.utils.data.DataLoader(
        temp_dataset,
        batch_size=conf[mode]['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=conf[mode]['drop_last'],
        num_workers=0,
        sampler=sampler
    )

    return dataloader, sampler

def load_mpose(dataset, split, verbose=False):
    dataset = mpose.MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess=None, 
                    velocities=True, 
                    remove_zip=False,
                    verbose=verbose)
    dataset.reduce_keypoints()
    dataset.scale_and_center()
    dataset.remove_confidence()
    dataset.flatten_features()
    
    return dataset.get_data()