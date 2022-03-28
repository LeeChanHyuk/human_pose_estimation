import torch
import torch.nn as nn
import torch.nn.functional as F
from .GAN_layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphAttentionLayer2


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adjacency_matrix, softmax_dim):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, softmax_dim=softmax_dim) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, softmax_dim = softmax_dim)
        self.softmax_dim = softmax_dim

    def forward(self, x, adj): # batch, sequence_length, 8, 4
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=3)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=3)


class GAT2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adjacency_matrix):
        """Dense version of GAT."""
        super(GAT2, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer2(nfeat, nhid, dropout=dropout, alpha=alpha, adjacency_matrix=adjacency_matrix, sequence_length=20, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer2(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adjacency_matrix=adjacency_matrix, sequence_length=20, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=3)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=3)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

