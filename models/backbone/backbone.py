from torch.nn import *

from models.backbone.cnn import CNN
from models.backbone.resnet import ResNet
from models.utils.weight_init import initialize_weights

model_pool = ('VGG11', 'VGG13', 'VGG16', 'VGG19',
              'ResNet-50', 'ResNet-101', 'ResNet-152',
              )


class BackBone (Module):

    def __init__(self, cfg):
        super(BackBone, self).__init__()
        self.cfg = cfg
        # 搭建主干网
        if 'VGG' in self.cfg.MODEL.BACKBONE.NAME:
            self.features = _make_CNN(cfg)
        elif 'ResNet' in self.cfg.MODEL.BACKBONE.NAME:
            self.features = _make_resnet(cfg)
        initialize_weights(self)

    def forward(self, x):
        return self.features(x)


def _make_CNN(cfg):
    return CNN(cfg)


def _make_resnet(cfg):
    return ResNet(cfg)
