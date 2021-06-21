r""" Convenience class for tracking the running average of data.
"""

import torch


class RunningAverage:
    def __init__(self, size):
        self.size = size
        self.average = torch.zeros(size)
        self.n = 0

    def add_item(self, item):
        r""" Add an new item to the running average.
        """
        self.average = (self.n * self.average + item) / (self.n + 1)
        self.n += 1

    def reset(self):
        self.average = torch.zeros(self.size)
        self.n = 0
