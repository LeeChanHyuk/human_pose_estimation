from distutils.command.build import build
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from template.trainer.architecture import action_transformer
import yaml
import numpy as np
import time

def build_model(num_classes=-1):
    with open("/media/ddl/새 볼륨/Git/human_pose_estimation/human_pose/template/conf/architecture/action_transformer.yaml") as f:
        list_doc = yaml.load(f.read(), Loader=yaml.FullLoader)
        order = list_doc['mode']
        architecture = action_transformer.ActionTransformer2(
            list_doc['ntoken'],
            list_doc['nhead'][order],
            list_doc['sequence_length'],
            list_doc['nlayers'][order],
            list_doc['dropout'][order],
            list_doc['mlp_size'][order],
            list_doc['classes']
        )
        model = architecture.to('cuda', non_blocking=True)
        #model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
        return model
    return None

def load_for_inference(rank, checkpoint_name = None):
    model = build_model()
    # For using DDP
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    base_path = os.getcwd()
    checkpoint_path = os.path.join(base_path, checkpoint_name)
    # Load state_dict
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def inference(pose_sequence):
    checkpoint_name = 'template/outputs/2022-02-13/00-38-37/checkpoint/top/001st_checkpoint_epoch_473.pth.tar'
    model = load_for_inference(0, checkpoint_name=checkpoint_name)
    pose_sequence = data_normalization(pose_sequence)
    pose_sequence = torch.Tensor(pose_sequence).to(device='cuda')
    start_time = time.time()
    y_pred = model(pose_sequence)
    #print('model fps = ', str(1/(time.time() - start_time)))
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.around(y_pred)
    y_pred = y_pred[0]
    if y_pred == 0:
        state = 'standard'
    elif y_pred == 1:
        state = 'Yaw+'
    elif y_pred == 2:
        state = 'Yaw-'
    elif y_pred == 3:
        state = 'Pitch+'
    elif y_pred == 4:
        state = 'Pitch-'
    elif y_pred == 5:
        state = 'Roll+'
    elif y_pred == 6:
        state = 'Roll-'
    return state

def data_normalization(data : np.array):
    for i in range(data.shape[-1]):
        data[:,i] = (data[:,i] - min(data[:,i])) / max(data[:,i])
    return data