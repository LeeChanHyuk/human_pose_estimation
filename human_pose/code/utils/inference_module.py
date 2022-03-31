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
    base_path = os.getcwd()
    with open(os.path.join(base_path, "C:/Users/user/Desktop/git/human_pose_estimation/human_pose/template/conf/architecture/action_transformer.yaml")) as f:
        list_doc = yaml.load(f.read(), Loader=yaml.FullLoader)
        order = list_doc['mode']
        architecture = action_transformer.ActionTransformer3(
            ntoken=list_doc['ntoken'],
            nhead=list_doc['nhead'][order],
            dropout=list_doc['dropout'][order],
            mlp_size=list_doc['mlp_size'][order],
            classes=list_doc['classes'],
            nlayers=list_doc['nlayers'][order],
            sequence_length=list_doc['sequence_length'],
            alpha = list_doc['alpha'],
            n_hid = list_doc['gat_output_dim'][order],
            softmax_dim=list_doc['softmax_dim']
        )
        model = architecture.to('cuda', non_blocking=True)
        #model = DDP(model, device_ids=[0], output_device=0, find_unused_parameters=True)
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


results = np.zeros((20), dtype=np.uint8)
checkpoint_name = 'C:/Users/user/Desktop/git/human_pose_estimation/human_pose/output/best_9354.pth.tar'
model = load_for_inference(0, checkpoint_name=checkpoint_name)

def inference(pose_sequence):
    action_vote = np.zeros((18), dtype=np.uint8)
    threshold = 0.90
    pose_sequence = torch.Tensor(pose_sequence).to(device='cuda')
    start_time = time.time()
    y_pred = model(pose_sequence)
    y_pred = torch.softmax(y_pred, dim=1)
    #print('model fps = ', str(1/(time.time() - start_time)))
    y_pred = y_pred.detach().cpu().numpy()
    probability = np.max(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.around(y_pred)
    y_pred = y_pred[0]

    # 판단 동작을 한 칸씩 뒤로 민다.
    for i in range(results.size-1):
        results[i] = results[i+1]
        action_vote[results[i]] += 1
    results[-1] = y_pred
    action_vote[results[-1]] += 1
    max_voted_action_val = np.max(action_vote)
    max_voted_action_class = np.argmax(action_vote)
    

    # 최신 탐지 동작을 list에 넣는다.

    actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+', 'left', 'left-up', 'up',
    'right-up', 'right', 'right-down', 'down', 'left-down', 'zoom-in', 'zoom-out','standard']
    if probability > threshold and (max_voted_action_val > 15 and y_pred == max_voted_action_class) or (max_voted_action_val > 10 and max_voted_action_class == 0):
        state = actions[max_voted_action_class]
    else:
        state = 'standard'
    return state

def data_normalization(data : np.array):
    for i in range(data.shape[-1]):
        min_max = max(data[:,i]) - min(data[:,i])
        data[:,i] = (data[:,i] - min(data[:,i])) / min_max
    return data