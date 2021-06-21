import torch
import torch.nn as nn


class DDI_dedicom(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(DDI_dedicom, self).__init__()
        self.out_dim = out_dim
        self.hidden_dim = args.hidden_dim
        self.in_dim = in_dim
        self.lin = nn.Linear(self.in_dim, self.hidden_dim)
        self.b1 = nn.BatchNorm1d(self.hidden_dim)
        self.R = torch.empty(self.hidden_dim, self.hidden_dim)
        self.R = nn.init.uniform_(self.R)
        self.R = nn.Parameter(data=self.R, requires_grad=True)
        self.D = torch.empty(self.out_dim, self.hidden_dim)
        self.D = nn.init.uniform_(self.D)
        self.D = nn.Parameter(data=self.D, requires_grad=True)

    def forward(self, data):
        h, _, _ = data.x, data.edge_index, data.edge_attr
        target_edge_index = data.target_edge_index

        x1 = h.index_select(0, target_edge_index[0])
        x2 = h.index_select(0, target_edge_index[1])
        inputs_row = self.lin(x1)
        inputs_col = self.lin(x2)
        inputs_row = self.b1(inputs_row)
        inputs_col = self.b1(inputs_col)
        product1 = torch.einsum("aj,bj->abj", [inputs_row, self.D])
        product2 = product1.matmul(self.R)
        product3 = product2 * self.D
        tmp = product3.permute(1, 0, 2)
        tmp1 = tmp.matmul(torch.t(inputs_col))
        tmp2 = tmp1.diagonal(dim1=1, dim2=2)
        output = tmp2.permute(1, 0)
        return torch.sigmoid(output)
