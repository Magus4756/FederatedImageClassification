# encode: utf-8

import torch


def lable_flip(label):
    new_l = (label + 3) % 10
    return torch.LongTensor(new_l)

