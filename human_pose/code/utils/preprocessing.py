import numpy as np

def data_preprocessing(data: np.array) -> np.array:
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
    normalized_poses = data_normalization(normalized_poses)
    return normalized_poses

def data_normalization(data : np.array) -> np.array:
    for i in range(data.shape[-1]):
        data[:, i] -= data[0, i]
    return data