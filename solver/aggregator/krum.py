# encode=utf-8

import copy
import torch


class Krum:
    """
    参照 "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    """
    def __init__(self, num_compromised):
        self.h_distance = []
        self._f = num_compromised
        return

    def krum(self, weights: list):
        vectors = copy.deepcopy(weights)
        n = len(weights) - self._f - 2
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
        distance.sort(dim=0)
        distance = distance[: n].sum(dim=0)
        sorted_idx = distance.argsort()
        chosen_idx = int(sorted_idx[0])
        krum_w = weights[chosen_idx]
        #  print(chosen_idx)
        # self.h_distance.append(distance)
        return krum_w, chosen_idx

    def __call__(self, weights: list):
        return self.krum(weights)[0]

'''
class MutiKrum:
    """
    Algorithm refer to "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    0 < choose <= 1 is selecting proportionally, choose > 1 is the selected number.
    """
    def __init__(self, num_compromised, choose=0.):
        self.h_distance = []
        self.h_choose = []
        self._f = num_compromised
        self.choose = choose
        return

    def __call__(self, weights: list, *args, **kwargs):
        vectors = copy.deepcopy(weights)
        # model params map to vectors
        for i, v in enumerate(vectors):
            for name in v:
                v[name] = v[name].reshape([-1]).type(torch.FloatTensor)
            vectors[i] = torch.cat(list(v.values()))

        distance = np.zeros([len(vectors), len(vectors)])
        for i, v_i in enumerate(vectors):
            for j, v_j in enumerate(vectors[i:]):
                temp = v_i - v_j
                distance[i][j + i] = distance[j + i][i] = torch.matmul(temp, temp.T)
        distance.sort(axis=0)
        distance = distance[: len(weights) - self._f - 2].sum(axis=0)
        sorted_idx = distance.argsort()
        c = self.choose
        if 0 <= self.choose <= 1.0:
            c = np.round(len(weights) * self.choose)
        n = int(min(max(c, 1), len(weights)))
        chosen_idx = sorted_idx[:n]
        chosen_w = [weights[i] for i in chosen_idx]
        print(sorted_idx[:n], [distance[i] for i in chosen_idx])
        krum_w = FedAvg()(chosen_w)
        # self.h_distance.append(distance)
        # self.h_choose.append(sorted_idx[:n])
        return krum_w
'''