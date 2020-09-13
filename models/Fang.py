from torch.nn import Module, Linear, Softmax, Sequential

from models.backbone.cnn import CNN
from models.utils.weight_init import initialize_weights


class FangClassification(Module):
    def __init__(self, cfg):
        super(FangClassification, self).__init__()
        self.cfg = cfg
        self.backbone = _make_back_bone(cfg)
        self.classifier = _make_classifier(cfg)
        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _make_back_bone(cfg):
    return CNN(cfg)


def _make_classifier(cfg):
    layers = []
    in_channel = cfg.MODEL.BACKBONE.OUT_CHANNELS
    for hidden_layer in cfg.MODEL.CLASSIFIER.HIDDEN_LAYERS:
        layer = Linear(in_features=in_channel, out_features=hidden_layer)
        layers.append(layer)
        in_channel = hidden_layer
    layer = Linear(in_channel, cfg.MODEL.CLASSIFIER.NUM_CLASSES)
    layers.append(layer)
    layers.append(Softmax(cfg.MODEL.CLASSIFIER.NUM_CLASSES))
    return Sequential(*layers)
