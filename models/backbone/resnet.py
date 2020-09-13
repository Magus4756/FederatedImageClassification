from torch.nn import *

from models.utils.weight_init import initialize_weights


class ResBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv2d(in_channels, out_channels[0], kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels[0])
        self.relu = ReLU(True)

        self.conv2 = Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = self.bn1

        self.conv3 = Conv2d(out_channels[0], out_channels[1], kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels[1])

        if in_channels != out_channels:
            self.downsample = _make_downsample(in_channels, out_channels[1], stride)
        initialize_weights(self)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.in_channels != self.out_channels:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out


def _make_downsample(in_channels, out_channels, stride):
    L = []
    L.append(Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
    L.append(BatchNorm2d(out_channels))
    return Sequential(*L)


class ResNet(Module):

    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.ResNet = _make_resnet(self)
        avg_in = int(cfg.MODEL.BACKBONE.INPUT_RESOLUTION * cfg.MODEL.RESNET.POOLER_SCALE[-1])
        self.avgpool = AvgPool2d(avg_in, stride=1)
        initialize_weights(self)

    def forward(self, x):
        x = self.ResNet(x)
        x = self.avgpool(x)
        return x


def _make_resnet(model):
    L = []

    # layer 1
    in_channels = model.cfg.MODEL.BACKBONE.INPUT_CHENNEL
    out_channels = 64
    con1 = Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    bn1 = BatchNorm2d(64)
    rl1 = ReLU(True)
    layer1 = [con1, bn1, rl1]
    L += layer1
    in_channels = out_channels
    out_channels = [out_channels, in_channels * 4]

    for layer_idx in range(1, 5):
        layer = []
        if layer_idx == 1:
            # resnet 的第2层先池化，所以卷积时 stride=1
            pooler = MaxPool2d(kernel_size=3, stride=2, padding=1)
            layer.append(pooler)
            layer.append(ResBlock(in_channels, out_channels))
        else:
            # 2-5层无池化，所以卷积时 stride=2
            # 第一个 block 通道数 * 4
            layer.append(ResBlock(in_channels, out_channels, stride=2))
        # 其余 block 通道数不变
        in_channels = out_channels[1]
        out_channels[0] = out_channels[1]
        for _ in range(1, model.cfg.MODEL.RESNET.BLOCK_NUM[layer_idx]):
            layer.append(ResBlock(in_channels, out_channels))
        L += layer
        in_channels = out_channels[1]
        out_channels = [int(in_channels / 2), in_channels * 2]

    # 计算当前特征图分辨率
    current_resolution = model.cfg.MODEL.RESNET.INPUT_RESOLUTION * model.cfg.MODEL.RESNET.POOLER_SCALE[4]
    assert int(current_resolution) == current_resolution, \
        'Input resolution and pooler scale are not match, please check your configure file.'
    kernel_size = int(current_resolution)
    # 把特征图转化为向量
    L += [AvgPool2d(kernel_size), ]
    return Sequential(*L)
