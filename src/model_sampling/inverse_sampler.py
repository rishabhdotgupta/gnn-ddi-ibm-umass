import torch
from model_sampling.sampling_impl import SamplingImpl


class InverseSampler(SamplingImpl):
    r""" Learns probabilities for each relation type.
    """
    def __init__(self, model, args, size):
        super(InverseSampler, self).__init__(model, args, size=size)

    def __get_logits__(self, data):
        r""" logit of an edge is the log of inverse frequency of its edge
        attributes
        """
        freq = data.edge_attr.sum(0) + 1
        fixed_logits = torch.log(1/freq)
        return torch.mv(data.edge_attr, fixed_logits)
