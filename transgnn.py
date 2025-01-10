import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, to_undirected
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj

class DegreePositionalEncoding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DegreePositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, degrees):
        return self.mlp(degrees)

class GNNMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(GNNMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_m = nn.Linear(nhead*d_model, d_model)

    def forward(self, x):
        q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_output = []
        for i in range(self.nhead):
            a_t = torch.matmul(q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
            a_t = F.softmax(a_t, dim=-1)
            h_i = torch.matmul(a_t, V)
            attn_output.append(h_i)

        attn_output = torch.cat(attn_output, dim=-1)
        attn_output = self.W_m(attn_output)
        return attn_output

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()
        self.self_attn = GNNMultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x
