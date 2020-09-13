# encode=utf-8

import random


class RandWeight:
    """
    随机选择一个
    """
    def __init__(self):
        return

    def __call__(self, weights):
        chosen = random.randint(0, len(weights) - 1)
        # print(chosen)
        return weights[chosen]
