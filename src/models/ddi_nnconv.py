import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class DDI_NNConv(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(DDI_NNConv, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = out_dim

        in_nn1 = nn.Sequential(nn.Linear(self.out_dim, 100), nn.ReLU(),
                               nn.Linear(100, self.in_dim * self.hidden_dim))
        self.conv1 = NNConv(self.in_dim, self.hidden_dim, nn=in_nn1)

        in_nn2 = nn.Sequential(
            nn.Linear(self.out_dim, 100), nn.ReLU(),
            nn.Linear(100, self.hidden_dim * self.hidden_dim))
        self.conv2 = NNConv(self.hidden_dim, self.hidden_dim, nn=in_nn2)

        self.bn = nn.BatchNorm1d(2 * self.hidden_dim)
        self.linear = nn.Linear(2 * self.hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        target_edge_index = data.target_edge_index

        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x1 = x.index_select(0, target_edge_index[0])
        x2 = x.index_select(0, target_edge_index[1])
        x = torch.cat([x1, x2], -1)
        x = self.bn(x)
        output = self.linear(x)
        return torch.sigmoid(output)
