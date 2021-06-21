import torch
from model_sampling.sampling_impl import SamplingImpl


class RandomSampler(SamplingImpl):
    r""" Learns probabilities for each relation type.
    """
    def __init__(self, model, args, size):
        super(RandomSampler, self).__init__(model, args, size=size)

    def __get_logits__(self, data):
        r""" All-zero logits will be random sampling
        """
        return torch.zeros(data.edge_index.size(1))
