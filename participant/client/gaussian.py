# encode=utf-8
from collections import OrderedDict

import torch

from .normal import Client


class GaussianClient(Client):
    def __init__(self, name, train_set, sample_set, logger, mean=0, variance=1):
        super(GaussianClient, self).__init__(name, train_set, sample_set, logger)
        self.compromise = True
        self.mean = mean
        self.variance = variance
        self.evil = 'gaussian'

    def train(self, lr=0.1, batch_size=0, epoch_num=1) -> [dict, float, float]:
        fake_w = OrderedDict()
        model_w = self.model.state_dict()
        for key in model_w:
            fake_w[key] = torch.randn(size=model_w[key].shape) * self.variance + self.mean
        for epoch in range(epoch_num):
            self.logger.logger.info('{} Train\tloss: {}\tacc: {}'.format(self, 0.0, 0.0))
        return fake_w, 0.0, 0.0