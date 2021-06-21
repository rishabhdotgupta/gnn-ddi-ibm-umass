import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from build_data import data_to_undirected
from running_average import RunningAverage
from torch_geometric.utils import remove_self_loops


class SamplingImpl(nn.Module):
    def __init__(self, model, args, size):
        super(SamplingImpl, self).__init__()
        self.model = model
        self.size = size
        self.device = args.device
        self.num_hops = args.model_depth
        out_dim = model.out_dim

        self.average_frac_sampled = RunningAverage(out_dim)
        self.average_sampled = RunningAverage(out_dim)
        self.subgraph_logit = 0

    @abc.abstractmethod
    def __get_logits__(self, data):
        return

    def ids_to_mask(self, ids, num_entries):
        mask = torch.zeros(num_entries).bool()
        mask[ids] = True
        return mask

    def __sample_mask__(self, to_sample, probs):
        return torch.multinomial(probs, to_sample, replacement=True)

    def __build__(self, data, edge_logits):
        r""" Build a subgraph using the edge logits.
        """
        # n_id[k] stores nodes in the boundary of the k-th hop.
        # It is initialized with the 0-th hop starting nodes.
        n_id = [data.target_edge_index.contiguous().view(-1).unique()]
        sample_mask = torch.zeros(edge_logits.size())
        sample_logit = 0

        for k in torch.arange(1, self.num_hops + 1):
            # find nodes adjacent to those in hop k-1
            src_nodes = data.edge_index[0]
            dest_nodes = data.edge_index[1]
            # act like graph is undirected.
            k_hop_mask = (src_nodes[:, None] == n_id[-1]).sum(1).bool()
            k_hop_mask += (dest_nodes[:, None] == n_id[-1]).sum(1).bool()

            if k_hop_mask.sum() > 0:
                probs = F.softmax(edge_logits[k_hop_mask], dim=0)
                to_sample = n_id[-1].size(0) * self.size[k-1]
                sample_ids = self.__sample_mask__(to_sample, probs)
                mask = self.ids_to_mask(sample_ids, probs.size(0)).float()
                sample_mask[k_hop_mask] += mask
                # track how likely it is to sample this graph
                sample_logit += edge_logits[k_hop_mask][sample_ids].sum()
                nodes = (data.edge_index[1] * sample_mask).long().unique()
                n_id.append(nodes)

        self.subgraph_logit = sample_logit
        return torch.clamp(sample_mask, 0, 1)

    def __sample_subgraph__(self, data):
        r""" Extract subsample from the subgraph surrounding
        data.target_edge_index
        """
        edge_logits = self.__get_logits__(data)
        sample_mask = self.__build__(data, edge_logits)
        edge_index = (data.edge_index * sample_mask).long()
        edge_attr = data.edge_attr * sample_mask[:, None]
        # remove 0->0 edges resulting from mask multiplication
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr,
                    y=data.y, target_edge_index=data.target_edge_index)

    def forward(self, data):
        data = data.to(torch.device('cpu'))
        sample = self.__sample_subgraph__(data)

        # print(f'{sample.edge_index.size(1)}/{data.edge_index.size(1)}')

        # track data on averages of sampling
        data_edge_freq = data.edge_attr.sum(0)
        indices = data_edge_freq.argsort(descending=True)
        sample_edge_freq = sample.edge_attr.sum(0)[indices]
        frac_sampled = sample_edge_freq / data_edge_freq[indices]
        self.average_sampled.add_item(sample_edge_freq)
        self.average_frac_sampled.add_item(frac_sampled)

        sample = data_to_undirected(sample)
        # return self.model(sample)
        return self.model(sample.to(self.device))
