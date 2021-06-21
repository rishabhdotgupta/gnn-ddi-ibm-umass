import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import coalesce
from build_data import data_to_undirected
from running_average import RunningAverage


class SigmoidSampling(nn.Module):
    def __init__(self, model, size, num_hops=2):
        super(SigmoidSampling, self).__init__()
        assert len(size) == num_hops, 'GumbelSampling: each hop needs size'
        self.model = model
        self.size = size
        self.num_hops = num_hops
        out_dim = model.out_dim

        self.average_frac_sampled = RunningAverage(out_dim)
        self.average_sampled = RunningAverage(out_dim)
        self.subgraph_logit = 0

    @abc.abstractmethod
    def __get_logits__(self, data):
        return

    # def __sample_mask__(self, to_sample, k_hop_logits):
    #     r""" Use gumbel-softmax to sample the k_hop neighbors.
    #     """
    #     gumbel_mask = torch.zeros(k_hop_logits.size())
    #     # gumbel_mask = torch.zeros(k_hop_logits.size()).to(self.device)
    #     for i in torch.arange(to_sample):
    #         gumbel_mask = gumbel_mask + F.gumbel_softmax(k_hop_logits,
    #                                                      tau=self.tau,
    #                                                      hard=True)
    #     gumbel_mask = torch.clamp(gumbel_mask, 0, 1)
    #     self.subgraph_logit = torch.dot(k_hop_logits, gumbel_mask)
    #
    #     return gumbel_mask

    def ids_to_mask(self, ids, num_entries):
        mask = torch.zeros(num_entries).bool()
        mask[ids] = True
        return mask

    def __sample_mask__(self, k_hop_logits):
        r""" Use gumbel-softmax to sample the k_hop neighbors.
        """
        probs = torch.sigmoid(k_hop_logits)
        sample_mask = torch.bernoulli(probs)
        return sample_mask

    def __build__(self, data, edge_logits):
        r""" Build a subgraph using the edge logits.
        """
        # n_id[k] stores nodes in the boundary of the k-th hop.
        # It is initialized with the 0-th hop starting nodes.
        n_id = [data.target_edge_index.contiguous().view(-1).unique()]
        # print("Number of nodes in 0th hop: %s" % len(n_id[-1]))
        e_id = []
        subgraph_logit = 0

        # source_nodes = data.edge_index[0]
        # # dest_nodes = data.edge_index[1]
        # # k_hop_mask = (src_nodes[:, None] == n_id[-1]).sum(1).bool()
        # # k_hop_mask += (dest_nodes[:, None] == n_id[-1]).sum(1).bool()
        # source_nodes_arg = source_nodes.numpy()
        # # edge_logits_arg = edge_logits.numpy()
        # edge_logits_arg = edge_logits.detach().numpy()
        #
        # for k in torch.arange(1, self.num_hops + 1):
        #     # find nodes adjacent to those in hop k-1
        #     if n_id[-1].size(0) > 0:
        #         n_id_arg = n_id[-1].numpy()
        #         # gumbel_mask = self.__sample_mask__(to_sample, edge_logits[k_hop_mask])
        #         sample_mask, k_hop_logit = nu.s(n_id_arg, self.size[k-1], source_nodes_arg, edge_logits_arg)
        #         subgraph_logit += k_hop_logit
        #         sample_mask = torch.tensor(sample_mask)

        for k in torch.arange(1, self.num_hops + 1):
            # find nodes adjacent to those in hop k-1
            src_nodes = data.edge_index[0]
            # dest_nodes = data.edge_index[1]
            k_hop_mask = (src_nodes[:, None] == n_id[-1]).sum(1).bool()
            # k_hop_mask = ((src_nodes[:, None] == n_id[-1]).sum(1) + (dest_nodes[:, None] == n_id[-1]).sum(1)).bool()
            if k_hop_mask.sum() > 0:
                gumbel_mask = self.__sample_mask__(edge_logits[k_hop_mask])
                # print(gumbel_mask.nonzero().shape)
                subgraph_logit += (F.logsigmoid(edge_logits[k_hop_mask])).sum() - torch.dot(edge_logits[k_hop_mask], 1-gumbel_mask)

                sample_mask = torch.zeros(k_hop_mask.size())
                sample_mask[k_hop_mask] = gumbel_mask
                e_id.append(sample_mask)
                nodes = (data.edge_index[1] * sample_mask).long().unique()
                n_id.append(nodes)
        self.subgraph_logit = subgraph_logit
        return e_id

    def __sample_subgraph__(self, data, edge_logits):
        r""" Extract subsample from the subgraph surrounding
        data.target_edge_index
        """
        e_id = self.__build__(data, edge_logits)

        # apply bitwise 'or' (a summation) over the edge_masks from each hop
        # to get the entire k-hop neighborhood
        sample_edge_masks = torch.stack(e_id, dim=0)
        sample_edge_mask = sample_edge_masks.sum(dim=0).clamp(0, 1)
        edge_index = (data.edge_index * sample_edge_mask).long()
        edge_attr = data.edge_attr * sample_edge_mask[:, None]
        # edge_attr = data.edge_attr[sample_edge_mask.bool()]

        # Using op=max because we want sampled edge_attr values to be 1
        edge_index, edge_attr = coalesce(edge_index, edge_attr,
                                         m=data.x.size(0),
                                         n=data.x.size(0),
                                         op='max')
        sample = Data(x=data.x, edge_index=edge_index,
                    edge_attr=edge_attr, y=data.y,
                    target_edge_index=data.target_edge_index)
        # print("Sample:", sample)
        sample = data_to_undirected(sample)
        edge_index, edge_attr = coalesce(sample.edge_index,
                                         sample.edge_attr,
                                         m=data.x.size(0),
                                         n=data.x.size(0),
                                         op='max')
        sample.edge_index, sample.edge_attr = edge_index[:, 1:], edge_attr[1:]
        return sample
        # return sample.to(self.device)

    def __sampler__(self, data):
        edge_logits = self.__get_logits__(data)
        return self.__sample_subgraph__(data, edge_logits)

    def forward(self, data):
        data = data.to(torch.device('cpu'))
        sample = self.__sampler__(data)

        # print(f'{sample.edge_index.size(1)}/{data.edge_index.size(1)}')

        # track data on averages of sampling
        data_edge_freq = data.edge_attr.sum(0)
        sample_edge_freq = sample.edge_attr.sum(0)
        frac_sampled = sample_edge_freq / data_edge_freq
        self.average_sampled.add_item(sample_edge_freq)
        self.average_frac_sampled.add_item(frac_sampled)

        # return self.model(sample)
        return self.model(sample.to(self.device))

