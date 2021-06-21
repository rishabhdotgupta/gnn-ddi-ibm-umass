import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class DDI_decagon(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(DDI_decagon, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = out_dim
        # self.device = args.device
        self.args = args

        # self.activation = F.relu
        self.dropout = nn.Dropout(p=0.1)

        self.R = torch.randn(self.hidden_dim, self.hidden_dim)
        self.R = nn.Parameter(data=self.R, requires_grad=True)
        self.D = torch.randn(self.out_dim, self.hidden_dim)
        self.D = nn.Parameter(data=self.D, requires_grad=True)

        self.build()

    def build(self):
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(RGCNConv(self.in_dim, self.hidden_dim, self.out_dim, self.args.basis))
        for i in range(self.args.model_depth - 1):
            self.gcn_layers.append(RGCNConv(self.hidden_dim, self.hidden_dim, self.out_dim, self.args.basis))

    def forward(self, data):
        h, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        target_edge_index = data.target_edge_index

        # # Implementation with random sampling
        edges, edge_type = torch.nonzero(edge_attr, as_tuple=True)
        edge_index = torch.index_select(edge_index, 1, edges)

        # Encoder
        # x = self.dropout(x)
        for layer in self.gcn_layers:
            h = layer(x=h, edge_index=edge_index, edge_type=edge_type)
            h = self.dropout(F.relu(h))
        x1 = h.index_select(0, target_edge_index[0])
        x2 = h.index_select(0, target_edge_index[1])

        # Decoder
        relation = torch.diag_embed(self.D)
        product1 = torch.matmul(x1, relation)
        product2 = torch.matmul(product1, self.R)
        product3 = torch.matmul(product2, relation)
        rec1 = torch.matmul(product3, torch.t(x2))
        rec1 = torch.diagonal(rec1, dim1=1, dim2=2)
        output = torch.t(rec1)
        return torch.sigmoid(output)
