import torch
import logging
import torchvision.models as torch_models
from torch import nn
from . import toy
LOGGER = logging.getLogger(__name__)
from efficientnet_pytorch import EfficientNet
from timm.models import create_model
from . import lstm
from . import action_transformer
from . import graph_neural_network
from .action_transformer_result_comparison import Video_action_transformer_v1, Video_action_transformer_v2
def create(conf, num_classes=None):
    base, architecture_name = [l.lower() for l in conf['type'].split('/')]
    print('model = ',base,architecture_name)
    if base == 'toy':
        architecture = toy.ToyModel()
    elif base == 'efficientv2':
        if architecture_name == 's': 
            architecture= create_model("tf_efficientnetv2_s",in_chans=3, num_classes=1)
        elif architecture_name == 'm': 
            architecture = create_model("tf_efficientnetv2_m",in_chans=3, num_classes=1)
        elif architecture_name == 'l': 
            architecture= create_model("tf_efficientnetv2_l",in_chans=1, num_classes=10)

    elif base == 'efficient':
        if architecture_name == 'b3': 
            architecture = EfficientNet.from_pretrained("efficientnet-b3", num_classes=1)
        elif architecture_name == 'b4': 
            architecture= EfficientNet.from_pretrained("efficientnet-b4", num_classes=1)
        elif architecture_name == 'b5': 
            architecture= EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        elif architecture_name == 'b6': 
            architecture = EfficientNet.from_pretrained("efficientnet-b6", num_classes=1)
        #architecture._conv_stem.in_channels = 40
        #totalweigh = torch.cat([architecture._conv_stem.weight[:,0:1],torch.cat([architecture._conv_stem.weight]*13,axis=1)],axis=1)
        #architecture._conv_stem.weight = torch.nn.Parameter(totalweigh)

    elif base == 'resnet':
        
        if architecture_name == '34': 
            architecture = torch_models.resnet34(True,{num_classes:1})
        elif architecture_name == '50': 
            architecture = torch_models.resnet50(True,{num_classes:1})
        elif architecture_name == '101': 
            architecture = torch_models.resnet101(True,{num_classes:1})
        architecture.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        architecture.fc = nn.Linear(2048,1)
    elif base == 'lstm':
        if architecture_name == 'x':
            architecture = lstm.LSTM(bidirection=False)
        elif architecture_name == 'bidirectional_lstm':
            architecture = lstm.LSTM(bidirection=True)
    elif base == 'action_transformer':
        if architecture_name == 'head_motion':
            order = conf['mode']
            architecture = action_transformer.ActionTransformer2(
                ntoken=conf['ntoken'],
                nhead=conf['nhead'][order],
                dropout=conf['dropout'][order],
                mlp_size=conf['mlp_size'][order],
                classes=conf['classes'],
                nlayers=conf['nlayers'][order],
                sequence_length=conf['sequence_length']
            ) # cls token 관련이 빠져있음. vector 중 0번째만 남기거나 이런게.
            print(conf['nhead'])
        elif architecture_name== 'gcn':
            order = conf['mode']
            architecture = action_transformer.ActionTransformer3(
                ntoken=conf['ntoken'],
                nhead=conf['nhead'][order],
                dropout=conf['dropout'][order],
                mlp_size=conf['mlp_size'][order],
                classes=conf['classes'],
                nlayers=conf['nlayers'][order],
                sequence_length=conf['sequence_length'],
                alpha = conf['alpha'],
                n_hid = conf['gat_output_dim'][order],
                softmax_dim=conf['softmax_dim']
            )
        elif architecture_name == 'test':
            order = conf['mode']
            architecture = action_transformer.ActionTransformer4(
                ntoken=conf['ntoken'],
                nhead=conf['nhead'][order],
                dropout=conf['dropout'][order],
                mlp_size=conf['mlp_size'][order],
                classes=conf['classes'],
                nlayers=conf['nlayers'][order],
                sequence_length=conf['sequence_length'],
                pose_node_num=conf['pose_node_num']
            )
    elif base == 'video_action_transformer':
        if architecture_name == 'v1':
            architecture = Video_action_transformer_v1.Semi_Transformer(conf['classes'], conf['sequence_length'])
        elif architecture_name == 'v2':
            architecture_name = Video_action_transformer_v2.Semi_Transformer(conf['classes'], conf['sequence_length'])
    else:
        raise AttributeError(f'not support architecture config: {conf}')

    return architecture