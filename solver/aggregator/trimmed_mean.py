# encode=utf-8

import copy
import torch

from .fed_avg import FedAvg


class TrimmedMean:
    """

    """
    def __init__(self, num_compromised):
        self._f = num_compromised
        return

    def __call__(self, weights: list):
        vectors = copy.deepcopy(weights)
        n = len(weights) - self._f - 1
        # model params map to vectors
        for i, v in enumerate(vectors):
            for name in v:
                v[name] = v[name].reshape([-1]).type(torch.FloatTensor)
            vectors[i] = torch.cat(list(v.values()))

        distance = torch.zeros([len(vectors), len(vectors)])
        for i, v_i in enumerate(vectors):
            for j, v_j in enumerate(vectors[i:]):
                temp = v_i - v_j
                distance[i][j + i] = distance[j + i][i] = torch.matmul(temp, temp.T)
        distance = distance.sum(dim=1)
        med = distance.median()
        _, chosen = torch.sort(abs(distance - med))
        chosen = chosen[: n]
        # print(chosen, distance)
        trimmed_mean = FedAvg()([weights[int(i)] for i in chosen])
        return trimmed_mean
