from re import A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, softmax_dim= 3):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 8)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features) # linear layer 한번 거치고
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # 
        attention = F.softmax(attention, dim=3)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) # node matrix와 weight matrix(a) 의 절반 간의 mul # 여기는 왜  wh와 wh간의 곱을 안구하고, 다른 애와의 곱을 구하는 거지?
        # 이런 방식을 통해 구한 애들은 node간의 연결성을 가지고 있다. 그 후 이 연결성과 인풋을 곱함으로써 연결성을 반영해주는 것이다.
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) # 이것도.
        Wh2 = Wh2.permute(0, 1, 3, 2)
        #Wh1 = self.expands(Wh1)
        #Wh2 = self.expands(Wh2)
        # broadcast add
        
        e = Wh1 + Wh2 # 더해서 정방행렬의 형태로 나타낸다.
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def expands(self, data: torch.tensor) -> torch.tensor:
        if data.shape[2] != data.shape[3]:
            if data.shape[2] > data.shape[3]:
                repeat_num = data.shape[2] // data.shape[3]
                data = torch.repeat_interleave(data, repeat_num, dim=3)
            else:
                repeat_num = data.shape[3] // data.shape[2]
                data = torch.repeat_interleave(data, repeat_num, dim=2)
        return data


class GraphAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, adjacency_matrix, sequence_length, concat=True):
        super(GraphAttentionLayer2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, adjacency_matrix.shape[0])))
        self.W2 = nn.Parameter(torch.empty(size=(in_features, adjacency_matrix.shape[0])))
        self.W3 = nn.Parameter(torch.empty(size=(adjacency_matrix.shape[0], self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 8)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.adaptive_A = nn.Parameter(torch.from_numpy(adjacency_matrix.astype(np.float32))).cuda() # 채널 별 adjacency matrix
        self.adaptive_A = self.adaptive_A.unsqueeze(dim=0)
        self.adaptive_A = torch.repeat_interleave(self.adaptive_A, sequence_length, dim=0)
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.silu = nn.SiLU()

    def forward(self, h): # h = (batch, sequence_length, node, node_features)
        Wh1 = torch.matmul(h, self.W) # wh1 = (batch, sequence_length, node, node_features)
        Wh2 = torch.matmul(h, self.W2) # wh2 = (batch, sequence_length, node, node_features) 
        Wh3 = self.tanh(Wh2 - Wh1)
        Wh4 = torch.matmul(Wh3, self.W3) # wh = (batch, sequence_length, node, out_features)
        e = self._prepare_attentional_mechanism_input(Wh4)

        adj = Wh3 * self.alpha + self.adaptive_A
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # 
        attention = F.softmax(attention, dim=3)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh4)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) # node matrix와 weight matrix(a) 의 절반 간의 mul # 여기는 왜  wh와 wh간의 곱을 안구하고, 다른 애와의 곱을 구하는 거지?
        # 이런 방식을 통해 구한 애들은 node간의 연결성을 가지고 있다. 그 후 이 연결성과 인풋을 곱함으로써 연결성을 반영해주는 것이다.
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) # 이것도.
        Wh2 = Wh2.permute(0, 1, 3, 2)
        Wh1 = self.expands(Wh1)
        Wh2 = self.expands(Wh2)
        # broadcast add
        
        e = Wh1 - Wh2 # 더해서 정방행렬의 형태로 나타낸다.
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def expands(self, data: torch.tensor) -> torch.tensor:
        if data.shape[2] != data.shape[3]:
            if data.shape[2] > data.shape[3]:
                repeat_num = data.shape[2] // data.shape[3]
                data = torch.repeat_interleave(data, repeat_num, dim=3)
            else:
                repeat_num = data.shape[3] // data.shape[2]
                data = torch.repeat_interleave(data, repeat_num, dim=2)
        return data

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
