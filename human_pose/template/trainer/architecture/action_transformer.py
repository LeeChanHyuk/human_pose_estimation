from turtle import forward
from unicodedata import bidirectional
from cv2 import repeat
from importlib_metadata import requires
import torch.nn as nn
import torch
from torch import Tensor
import math
from . import transformer
from torchvision import transforms

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


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
