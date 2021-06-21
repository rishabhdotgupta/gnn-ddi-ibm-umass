import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rgcn_prob import RGCNConv_prob


class DDI_decagon_prob(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(DDI_decagon_prob, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim1 = args.hidden_dim
        self.hidden_dim2 = self.hidden_dim1
        self.out_dim = out_dim
        self.device = args.device

        self.probs = nn.Parameter(torch.randn(self.out_dim),
                                  requires_grad=True)
        self.act = args.activation

        self.conv1 = RGCNConv_prob(in_dim, self.hidden_dim1, self.out_dim,
                                   args.basis, probs=self.probs, act=self.act)

        self.conv2 = RGCNConv_prob(self.hidden_dim1, self.hidden_dim2,
                                   self.out_dim, args.basis, probs=self.probs,
                                   act=self.act)

        self.dropout = nn.Dropout(p=0.1)

        self.R = torch.randn(self.hidden_dim2, self.hidden_dim2)
        self.R = nn.Parameter(data=self.R, requires_grad=True)
        self.D = torch.randn(self.out_dim, self.hidden_dim2)
        self.D = nn.Parameter(data=self.D, requires_grad=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        target_edge_index = data.target_edge_index

        # # Implementation with random sampling
        edges, edge_type = torch.nonzero(edge_attr, as_tuple=True)
        edge_index = torch.index_select(edge_index, 1, edges)

        # Encoder
        x = self.dropout(x)
        h = self.conv1(x=x, edge_index=edge_index, edge_type=edge_type)
        h = self.dropout(F.relu(h))
        h = self.conv2(x=h, edge_index=edge_index, edge_type=edge_type)
        h = F.relu(h)
        x1 = h.index_select(0, target_edge_index[0])
        x2 = h.index_select(0, target_edge_index[1])
        # x1 = self.bn(x1)
        # x2 = self.bn(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        # Decoder
        relation = torch.diag_embed(self.D)
        product1 = torch.matmul(x1, relation)
        product2 = torch.matmul(product1, self.R)
        product3 = torch.matmul(product2, relation)
        rec1 = torch.matmul(product3, torch.t(x2))
        rec1 = torch.diagonal(rec1, dim1=1, dim2=2)
        output = torch.t(rec1)
        return torch.sigmoid(output)
