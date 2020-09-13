# encode=utf-8

from torch.nn import *

from models.utils.weight_init import initialize_weights


class SingleLayer(Module):

    def __init__(self, cfg, init_weights=True):
        super(SingleLayer, self).__init__()
        self.cfg = cfg
        self.layer1 = Linear(784, 200)
        self.dropout = Dropout(0.5)
        self.layer2 = Linear(200, 10)
        self.ReLU = ReLU(inplace=True)
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        return x


class MNIST_2NN(Module):
    def __init__(self, cfg, init_weights=True):
        super(MNIST_2NN, self).__init__()
        self.cfg = cfg
        self.layer1 = Linear(784, 200)
        self.layer2 = Linear(200, 200)
        self.dropout = Dropout(0.5)
        self.layer3 = Linear(200, 10)
        self.ReLU = ReLU(inplace=True)
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.ReLU(x)
        return x


class CNN_2_Layer(Module):
    def __init__(self, cfg, init_weights=True):
        super(CNN_2_Layer, self).__init__()
        self.cfg = cfg
        self.CNN = self._make_cnn()
        self.FCN = self._make_fcn()
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        x = self.CNN(x)
        x = x.reshape([x.shape[0], -1])
        x = self.FCN(x)
        return x

    def _make_cnn(self):
        layers = []
        layers.append(Conv2d(in_channels=1, out_channels=32,  kernel_size=5, stride=1, padding=2))
        layers.append(MaxPool2d(kernel_size=2, stride=2))
        layers.append(Conv2d(in_channels=32, out_channels=64,  kernel_size=5, stride=1, padding=2))
        layers.append(MaxPool2d(kernel_size=2, stride=2))
        return Sequential(*layers)

    def _make_fcn(self):
        layers = []
        layers.append(Linear(3136, 512))
        layers.append(Linear(512, 10))
        layers.append(ReLU(inplace=True))
        layers.append(Softmax(dim=1))
        return Sequential(*layers)


class MLP(Module):
    def __init__(self, cfg, dim_hidden=64):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.layer_input = Linear(cfg.MODEL.BACKBONE.INPUT_RESOLUTION ** 2, dim_hidden)
        self.relu = ReLU()
        self.dropout = Dropout()
        self.layer_hidden = Linear(dim_hidden, cfg.MODEL.CLASSIFIER.NUM_CLASSES)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)
