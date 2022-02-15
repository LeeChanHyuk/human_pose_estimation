import numpy as np

def data_normalization(data : np.array):
    for i in range(data.shape[-1]):
        data[:,i] = (data[:,i] - min(data[:,i])) / (max(data[:,i] - min(data[:,i])))
    return data

def data_preprocessing(data: np.array) -> np.array:
    normalized_pose = []
    for i in range(data.shape[-1]):
        i_all = data[:,i]
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
    normalized_pose = data_normalization(normalized_pose)
    return normalized_pose