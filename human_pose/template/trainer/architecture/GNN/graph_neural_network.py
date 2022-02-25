from mimetypes import init
from turtle import forward
from unicodedata import bidirectional
from cv2 import _InputArray_KIND_MASK, repeat
from importlib_metadata import requires
import torch.nn as nn
import torch
from torch import Tensor
import math
from . import transformer
from torchvision import transforms

class GNN_Layer(nn.Module):
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.A = A
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(torch._sparse_mm(self.A, x))

class GCN(nn.Module):
    def __init__(self, num_features, num_class, A):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            GNN_Layer(num_features, 16, A),
            nn.ReLU(),
            GNN_Layer(16, num_class, A)
        )
    def forward(self, x):
        return self.feature_extractor(x)

