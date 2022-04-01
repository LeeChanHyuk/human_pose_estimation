from turtle import forward, pos
from unicodedata import bidirectional
from cv2 import repeat
from importlib_metadata import requires
import torch.nn as nn
import torch
from torch import Tensor
import math
from . import transformer
from torchvision import transforms
from .GNN.GAN_models import GAT, GAT2
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.CLS_Token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.positional_embedding = nn.Embedding(self.sequence_length+1, self.d_model)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.sequence_length+1, self.d_model))
        self.positions = torch.arange(start=0, end=self.sequence_length+1, dtype=torch.long, device='cuda')
        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.CLS_Token.data.uniform_(-initrange, initrange)
        self.positional_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        cls_tokens = self.CLS_Token.expand(x.shape[0], -1, -1).requires_grad_()
        x = torch.cat((cls_tokens, x), dim=1)
        positional_embeddings = self.positional_embedding(self.positions)
        x += positional_embeddings
        return x

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    
class ActionTransformer1(nn.Module):

    def __init__(self, ntoken: int, nhead: int, sequence_length: int,
                 nlayers: int, dropout: float = 0.5, mlp_size: int = 256, classes: int = 7):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 64 * nhead
        self.d_hid = self.d_model * 4
        self.positional_encoder = PositionalEmbedding(self.d_model, sequence_length)
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, self.d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.temp_encoder = nn.Linear(ntoken, self.d_model)
        self.encoder = nn.Linear(ntoken, self.d_model)
        self.dense_layer1 = nn.Linear(self.d_model, mlp_size)
        self.dense_layer2 = nn.Linear(mlp_size, classes)
        self.lambda_function = transforms.Lambda(lambd=lambda x: x[:,0,:])
        self.init_weights()
        

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.temp_encoder.weight.data.uniform_(-initrange, initrange)
        self.dense_layer1.weight.data.uniform_(-initrange, initrange)
        self.dense_layer2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # CLS Token만 추가 필요
 #       src = self.temp_encoder(src) * math.sqrt(self.d_model)
        x = self.encoder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = self.lambda_function(x)
        #encoder_output = encoder_output[:,0,:]
        x = self.dense_layer1(x)
        output = self.dense_layer2(x)
        #print(self.encoder.weight)
        return output

class ActionTransformer2(nn.Module):

    def __init__(self, ntoken: int, nhead: int, sequence_length: int,
                 nlayers: int, dropout: float = 0.5, mlp_size: int = 256, classes: int = 7):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 64 * nhead
        self.d_hid = self.d_model * 4
        self.positional_encoder = PositionalEmbedding(self.d_model, sequence_length)
        self.transformer_encoder = transformer.Encoder(self.d_model, self.d_hid, nhead, nlayers, drop_prob=dropout, device='cuda')
        self.temp_encoder = nn.Linear(ntoken, self.d_model)
        self.encoder = nn.Linear(ntoken, self.d_model)
        self.dense_layer1 = nn.Linear(self.d_model, mlp_size)
        self.dense_layer2 = nn.Linear(mlp_size, classes)
        self.lambda_function = transforms.Lambda(lambd=lambda x: x[:,0,:])
        self.init_weights()
        

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.temp_encoder.weight.data.uniform_(-initrange, initrange)
        self.dense_layer1.weight.data.uniform_(-initrange, initrange)
        self.dense_layer2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # CLS Token만 추가 필요
 #       src = self.temp_encoder(src) * math.sqrt(self.d_model)
        x = self.encoder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, None)
        x = self.lambda_function(x)
        #encoder_output = encoder_output[:,0,:]
        x = self.dense_layer1(x)
        output = self.dense_layer2(x)
        #print(self.encoder.weight)
        return output

class ActionTransformer3(nn.Module):

    def __init__(self, ntoken: int, nhead: int, sequence_length: int,
                 nlayers: int, dropout: float = 0.5, mlp_size: int = 256, classes: int = 7, alpha=0.3, n_hid = 32, softmax_dim = 3):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 64 * nhead
        self.d_hid = self.d_model * 4
        self.positional_encoder = PositionalEmbedding(self.d_model, sequence_length)
        self.transformer_encoder = transformer.Encoder(self.d_model, self.d_hid, nhead, nlayers, 0.1, 'cuda')
        self.temp_encoder = nn.Linear(ntoken, self.d_model)
        self.encoder = nn.Linear(self.d_model * 8, self.d_model)
        self.dense_layer1 = nn.Linear(self.d_model, mlp_size)
        self.dense_layer2 = nn.Linear(mlp_size, classes)
        self.lambda_function = transforms.Lambda(lambd=lambda x: x[:,0,:])
        self.adjacency_matrix = torch.from_numpy(np.array(
            [[1,1,0,0,0],
            [1,1,1,1,0],
            [0,1,1,1,1],
            [0,1,1,1,1],
            [0,0,1,1,1]]
        )).cuda()
        self.adjacency_matrix2 = torch.from_numpy(np.array(
        [[1,1,0,0,0,0,1,1],
        [1,1,1,1,0,1,1,0],
        [0,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,1,1,1,1,1,0,0],
        [1,1,0,0,0,0,1,1],
        [1,0,0,0,0,0,1,1]]
        )).cuda()
        self.adjacency_matrix3 = np.array(
        [[1,1,0,0,0,0,1,1],
        [1,1,1,1,0,1,1,0],
        [0,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,1,1,1,1,1,0,0],
        [1,1,0,0,0,0,1,1],
        [1,0,0,0,0,0,1,1]]
        )
        self.transition_batch_normalization = nn.BatchNorm1d(20)
        self.pose_batch_normalization = nn.BatchNorm1d(20)
        self.silu = nn.SiLU()
        self.body_pose_embedding = nn.Embedding(num_embeddings=37*37*37, embedding_dim = 10)
        self.head_pose_embedding = nn.Embedding(num_embeddings=37*37*37, embedding_dim = 10)
        self.eye_pose_embedding = nn.Embedding(num_embeddings=21*21*21*21, embedding_dim = 10)

        # encoder
        self.pose_encoder = nn.Linear(3, 12)
        self.transition_encoder = nn.Linear(15, 20)
        """GAT Hyper parameters
        alpha = 0.1, 0.2, 0.3
        nhid = 16, 32, 64
        """
        self.GAT = GAT(nfeat=4,nhid=n_hid, nclass=self.d_model, dropout=0.1, alpha=0.9, nheads=3, adjacency_matrix=self.adjacency_matrix2, softmax_dim=softmax_dim).to('cuda')
        #self.GAT2 = GAT2(nfeat=4,nhid=32, nclass=self.d_model, dropout=0.1, alpha=alpha, nheads=3, adjacency_matrix=self.adjacency_matrix3).to('cuda')
        self.init_weights()
        

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.temp_encoder.weight.data.uniform_(-initrange, initrange)
        self.dense_layer1.weight.data.uniform_(-initrange, initrange)
        self.dense_layer2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # CLS Token만 추가 필요
 #       src = self.temp_encoder(src) * math.sqrt(self.d_model)
        # (x = 5x3)
        transitions = x[:,:,:15]
        transitions = self.transition_encoder(transitions)
        transitions = self.transition_batch_normalization(transitions)
        transitions = self.silu(transitions)

        postures = x[:,:,15:]
        postures = self.pose_encoder(postures)
        postures = self.pose_batch_normalization(postures)
        postures = self.silu(postures)
        
        feature_matrix = torch.cat([transitions[:,:,0:4].unsqueeze(2), transitions[:,:,4:8].unsqueeze(2), transitions[:,:,8:12].unsqueeze(2), transitions[:,:,12:16].unsqueeze(2),transitions[:,:,16:20].unsqueeze(2), postures[:,:,0:4].unsqueeze(2), postures[:,:,4:8].unsqueeze(2), postures[:,:,8:12].unsqueeze(2)], dim=2)
        x = self.GAT(feature_matrix, self.adjacency_matrix2)
        #x = self.GAT2(feature_matrix)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.silu(x)
        x = self.encoder(x)
        x = self.silu(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, None)
        x = self.lambda_function(x)
        #encoder_output = encoder_output[:,0,:]
        x = self.dense_layer1(x)
        x = self.silu(x)
        output = self.dense_layer2(x)
        output = self.silu(output)
        #print(self.encoder.weight)
        return output

class ActionTransformer4(nn.Module):

    def __init__(self, ntoken: int, nhead: int, sequence_length: int,
                 nlayers: int, dropout: float = 0.5, mlp_size: int = 256, classes: int = 7, pose_node_num = 5):
        super().__init__()
        
        # basic
        self.model_type = 'Transformer'
        self.d_model = 64 * nhead
        self.d_hid = self.d_model * 4
        self.positional_encoder = PositionalEmbedding(self.d_model, sequence_length)
        self.transformer_encoder = transformer.Encoder(self.d_model, self.d_hid, nhead, nlayers, drop_prob=dropout, device='cuda')
        self.lambda_function = transforms.Lambda(lambd=lambda x: x[:,0,:])
        self.pose_batch_normalization = nn.BatchNorm1d(20)
        self.transition_batch_normalization = nn.BatchNorm1d(20)
        self.cls_token_batch_normlaization = nn.BatchNorm1d(1)
        self.dense_layer_batch_normalization = nn.BatchNorm1d(1)

        # encoder
        self.pose_encoder = nn.Linear(10, int(self.d_model/2))
        self.transition_encoder = nn.Linear(15, int(self.d_model/2))
        self.combine_encoder = nn.Linear(int(self.d_model), self.d_model)
        self.temp_encoder = nn.Linear(ntoken, self.d_model)
        self.encoder = nn.Linear(ntoken, self.d_model)
        self.dense_layer1 = nn.Linear(self.d_model, mlp_size)
        self.dense_layer2 = nn.Linear(mlp_size, classes)
        
        # attention
        self.pose_attention = transformer.MultiHeadAttention(int(self.d_model / 2), nhead)
        self.transition_attention = transformer.MultiHeadAttention(int(self.d_model/2), nhead)
        self.cls_token_attention = transformer.MultiHeadAttention(self.d_model, int(self.d_model/2))
        self.init_weights()

        

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.temp_encoder.weight.data.uniform_(-initrange, initrange)
        self.dense_layer1.weight.data.uniform_(-initrange, initrange)
        self.dense_layer2.weight.data.uniform_(-initrange, initrange)
        self.pose_encoder.weight.data.uniform_(-initrange, initrange)
        self.transition_encoder.weight.data.uniform_(-initrange, initrange)
        self.combine_encoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # CLS Token만 추가 필요
 #       src = self.temp_encoder(src) * math.sqrt(self.d_model)
        poses = self.pose_encoder(x[:,:,15:])
        poses = self.pose_batch_normalization(poses)
        transitions = self.transition_encoder(x[:,:,:15])
        transitions = self.transition_batch_normalization(transitions)
        x = torch.cat([transitions, poses], dim=2)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, None)
        x = self.lambda_function(x)
        #encoder_output = encoder_output[:,0,:]
        x = torch.unsqueeze(x, dim=1)
        x = self.cls_token_attention(q=x, k=x, v=x, mask = None)
        x = self.dense_layer1(x)
        output = self.dense_layer2(x)
        #print(self.encoder.weight)
        output = torch.squeeze(output,dim=1)
        return output

class LookingClassifier(nn.Module):

    def __init__(self, ntoken: int, nhead: int, sequence_length: int,
                 nlayers: int, dropout: float = 0.5, mlp_size: int = 256, classes: int = 7, pose_node_num = 5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 64 * nhead
        self.d_hid = self.d_model * 4
        self.positional_encoder = PositionalEmbedding(self.d_model, sequence_length)
        self.transformer_encoder = transformer.Encoder(self.d_model, self.d_hid, nhead, nlayers, 0.1, 'cuda')
        self.temp_encoder = nn.Linear(ntoken, self.d_model)
        self.encoder = nn.Linear(ntoken+pose_node_num, self.d_model)
        self.dense_layer1 = nn.Linear(self.d_model, mlp_size)
        self.dense_layer2 = nn.Linear(mlp_size, classes)
        self.pose_encoder = nn.Linear(10, int(self.d_model/2))
        self.transition_encoder = nn.Linear(15, int(self.d_model/2))
        self.combine_encoder = nn.Linear(int(self.d_model), self.d_model)
        self.lambda_function = transforms.Lambda(lambd=lambda x: x[:,0,:])
        self.pose_attention = transformer.MultiHeadAttention(int(self.d_model / 2), nhead)
        self.transition_attention = transformer.MultiHeadAttention(int(self.d_model/2), nhead)
        self.init_weights()
        

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.temp_encoder.weight.data.uniform_(-initrange, initrange)
        self.dense_layer1.weight.data.uniform_(-initrange, initrange)
        self.dense_layer2.weight.data.uniform_(-initrange, initrange)
        self.pose_encoder.weight.data.uniform_(-initrange, initrange)
        self.transition_encoder.weight.data.uniform_(-initrange, initrange)
        self.combine_encoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # CLS Token만 추가 필요
 #       src = self.temp_encoder(src) * math.sqrt(self.d_model)
        poses = self.pose_encoder(x[:,:,15:])
        transitions = self.transition_encoder(x[:,:,:15])
        x = torch.cat([transitions, poses], dim=2)
        x = self.combine_encoder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, None)
        x = self.lambda_function(x)
        #encoder_output = encoder_output[:,0,:]
        x = self.dense_layer1(x)
        output = self.dense_layer2(x)
        #print(self.encoder.weight)
        return output
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
