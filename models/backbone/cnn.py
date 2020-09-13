from torch.nn import *


class CNN(Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()

        layers = []
        in_channels = cfg.MODEL.BACKBONE.INPUT_CHENNEL  # 保存每层的输出数作为下一层的输入数
        pooler_scale = 1  # 保存每一层池化的比例，计算下一层
        for idx, (out_channels, l) in enumerate(cfg.MODEL.CNN.CONV):
            # 新建卷积层
            for _ in range(l):
                conv2d = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg.MODEL.CNN.KERNEL_SIZE,
                    padding=1
                )
                layers += [conv2d, BatchNorm2d(out_channels), ReLU(inplace=True)]
                in_channels = out_channels
            # 新建池化层
            scale = int(pooler_scale / cfg.MODEL.CNN.POOLER_SCALE[idx])
            pool = MaxPool2d(
                kernel_size=cfg.MODEL.CNN.POOLER_SIZE,
                stride=scale
            )
            pooler_scale = cfg.MODEL.CNN.POOLER_SCALE[idx]
            layers += [pool]

        self.features = Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x
