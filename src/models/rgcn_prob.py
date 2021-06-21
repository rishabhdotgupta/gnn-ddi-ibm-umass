import torch
import torch.nn.functional as f
from torch_geometric.nn.conv import RGCNConv


class RGCNConv_prob(RGCNConv):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 probs, act, root_weight=True, bias=True, **kwargs):
        super(RGCNConv_prob, self).__init__(in_channels, out_channels,
                                            num_relations, num_bases)
        self.probs = probs
        self.act = act

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            p = torch.index_select(self.probs, 0, edge_type)
            if self.act == 'softmax':
                p = f.softmax(p, dim=0)
            else:
                p = torch.sigmoid(p)
            out = (p.unsqueeze(1))*out

        return out if edge_norm is None else out * edge_norm.view(-1, 1)
