
from torch.nn import *


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, Conv2d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
