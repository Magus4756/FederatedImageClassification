# encode=utf-8

import copy


class FedAvg:
    """
    计算参数平均值
    """
    def __init__(self):
        return

    def __call__(self, weights: list):
        if len(weights) == 1:
            return weights[0]
        avg_w = copy.deepcopy(weights[0])
        for key in avg_w:
            for w in weights[1:]:
                avg_w[key] += w[key]
            avg_w[key] /= len(weights)
        return avg_w
