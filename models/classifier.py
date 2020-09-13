import copy

from torch.nn import *

from models.utils.weight_init import initialize_weights


class Classifier(Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.drop = Dropout(0.5)
        self.FCN = _make_FCN(cfg)
        initialize_weights(self)

    def forward(self, x):
        x = self.drop(x)
        x = self.FCN(x)
        return x


def _make_FCN(cfg):
    in_channel = cfg.MODEL.BACKBONE.OUT_CHENNELS
    out_channel = cfg.MODEL.CLASSIFIER.NUM_CLASSES
    layer = Linear(in_channel, out_channel)
    return layer
