from cv2 import split
from matplotlib.pyplot import axis
import numpy as np
from torch import normal

def data_preprocessing(data: np.array, fps) -> np.array:
    poses_from_one_video = data
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
    #body_poses = poses_from_one_video[6, :, :3] / 90
    #gaze_poses = poses_from_one_video[7]
    all_poses = np.concatenate([center_eyes, center_mouths, left_shoulders, right_shoulders, center_stomachs, head_poses], axis=1) # (frames, values * 6)
    normalized_poses = []
    for i in range(all_poses.shape[-1]):
        i_all = all_poses[:,i]
        i_all = size_normalization(i_all, fps)

        normalized_poses.append(i_all)
    normalized_poses = np.array(normalized_poses).transpose()
    normalized_poses = data_normalization(normalized_poses)
    return normalized_poses

def data_normalization(data : np.array) -> np.array:
    for i in range(data.shape[-1]):
        data[:, i] -= data[0, i]
    return data

def size_normalization(data: np.array, fps: int) -> np.array: # input shape = (sequence_length, 1)
    normalize_num = int(data.shape[0] - 20)
    if normalize_num < 0: # Append the feature in last frame to every feature. # if the frame rate is under than 10.
        for i in range(abs(normalize_num)):
            appended_data = np.array(data[-1])
            data = np.append(data, np.array([data[-1]]), axis=-1)
        return data
    if fps == 10:
        return data
    elif fps < 20:
        normalize_num = 2 * (data.shape[0] - 20)
        temp_data = data[-normalize_num:]
        temp_data = (data[0:len(temp_data):2] + data[1:len(temp_data):2]) / 2
        new_array = np.concatenate([data[0:len(data) - normalize_num], temp_data])
    else: # fps > 20:
        new_array = data
        while len(new_array) > 40:
            if len(new_array) % 2:
                new_array = new_array[0:len(new_array)-1]
            data_1 = new_array[0:len(new_array):2]
            data_2 = new_array[1:len(new_array):2]
            new_array = (data_1 + data_2) / 2
        if len(new_array) % 2:
            new_array = new_array[0:len(new_array)-1]
        normalize_num = 2 * (new_array.shape[0] - 20)
        temp_data = new_array[-normalize_num:]
        temp_data = (temp_data[0:len(temp_data):2] + temp_data[1:len(temp_data):2]) / 2
        new_array = np.concatenate([new_array[0:len(new_array) - normalize_num], temp_data])
    if new_array.shape[0] > 20:
        print('call')
    return new_array