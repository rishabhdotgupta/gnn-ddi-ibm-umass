import torch
import torch.nn as nn


class DDI_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(DDI_MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = out_dim
        self.args = args

        self.classifer = nn.Sequential(*self.build())

    def build(self):
        layers = []
        layers.append(nn.Linear(2 * self.in_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        for i in range(self.args.model_depth - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return layers

    def forward(self, data):
        h, target_edge_index = data.x, data.target_edge_index

        h1 = h.index_select(0, target_edge_index[0])
        h2 = h.index_select(0, target_edge_index[1])
        h3 = torch.cat([h1, h2], -1)
        output = self.classifer(h3)
        return torch.sigmoid(output)
