from audioop import minmax
from distutils.command.build import build
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from template.trainer.architecture import action_transformer
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import numpy as np
import time

def build_model(num_classes=-1):
    os.environ['MASTER_ADDR'] = '127.0.0.3'
    os.environ['MASTER_PORT'] = '9095'
    dist.init_process_group(
        backend='nccl', world_size=1, init_method='env://',
        rank=0
    )
    base_path = os.getcwd()
    with open(os.path.join(base_path, "template/conf/architecture/action_transformer.yaml")) as f:
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
        model = DDP(model, device_ids=[0], output_device=0, find_unused_parameters=True)
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
    model.module.load_state_dict(checkpoint['model'])
    model.eval()
    return model

checkpoint_name = '/media/ddl/새 볼륨/Git/human_pose_estimation/human_pose/outputs/2022-02-21/22-28-08/checkpoint/top/001st_checkpoint_epoch_77.pth.tar'
model = load_for_inference(0, checkpoint_name=checkpoint_name)

def inference(pose_sequence):
    results = [0, 0, 0]
    results[-3] = results[-2]
    results[-2] = results[-1]
    threshold = 0.5
    pose_sequence = torch.Tensor(pose_sequence).to(device='cuda')
    start_time = time.time()
    y_pred = model(pose_sequence)
    y_pred = torch.softmax(y_pred, dim=1)
    #print('model fps = ', str(1/(time.time() - start_time)))
    y_pred = y_pred.detach().cpu().numpy()
    probability = np.max(y_pred)
    print(probability)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.around(y_pred)
    y_pred = y_pred[0]
    results[-1] = y_pred
    actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+', 'left', 'left_up', 'up',
    'right_up', 'right', 'right_down', 'down', 'left_down', 'zoom_in', 'zoom_out','standard']
#    if probability > threshold and (results[-1] == results[-2]) and (results[-2] == results[-3]):
    if probability > threshold:
        state = actions[y_pred]
    else:
        state = 'None'
    print(state)
    return state

def data_normalization(data : np.array):
    for i in range(data.shape[-1]):
        min_max = max(data[:,i]) - min(data[:,i])
        data[:,i] = (data[:,i] - min(data[:,i])) / min_max
    return data