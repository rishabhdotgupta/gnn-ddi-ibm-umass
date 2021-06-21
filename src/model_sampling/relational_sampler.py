import torch
import torch.nn as nn
from model_sampling.sampling_impl import SamplingImpl


class RelationalSampler(SamplingImpl):
    r""" Learns probabilities for each relation type.
    """
    def __init__(self, model, args, size):
        super(RelationalSampler, self).__init__(model, args, size=size)
        # number of probabilities is the number of relations to predict
        self.np = self.model.out_dim
        self.edge_params = nn.Parameter(torch.randn(self.np),
                                        requires_grad=True)

    def __get_logits__(self, data):
        r""" Compute Av. These values are interpreted as logits
        """
        # return torch.mv(data.edge_attr, self.edge_params)
        edge_params = self.edge_params.to(torch.device('cpu'))
        return torch.mv(data.edge_attr, edge_params)
