from torch import nn
from dpipe.layers import make_consistent_seq, ResBlock2d, Reshape, PostActivation2d

from defconv.defconv import DefConv


class ConvNet(nn.Module):
    def __init__(self, in_channels, channels, n_classes, kernel_size=3, padding=0, stride=1,
                 pool_kernel=1):
        super().__init__()
        self.n_classes = n_classes

        self.net = make_consistent_seq(PostActivation2d, channels=[in_channels] + channels,
                                       kernel_size=kernel_size,
                                       padding=padding, stride=stride)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(pool_kernel),
                                Reshape('0', -1),
                                nn.Linear(channels[-1], n_classes)
                                )

    def forward(self, x):
        x = self.net(x)
        return nn.Softmax(dim=1)(self.fc(x))


class DeformConvNet(nn.Module):
    def __init__(self, in_channels, channels, n_classes, kernel_size=3, padding=0, stride=1,
                 pool_kernel=1):
        super(DeformConvNet, self).__init__()

        self.n_classes = n_classes

        self.net = nn.Sequential(
            PostActivation2d(in_channels, channels[0], kernel_size=kernel_size,
                             padding=padding, stride=stride),

            DefConv(channels[1]),
            PostActivation2d(channels[1], channels[2], kernel_size=kernel_size,
                             padding=padding, stride=stride),
            PostActivation2d(channels[2], channels[2], kernel_size=kernel_size,
                             padding=padding, stride=stride),
            DefConv(channels[3]),
            PostActivation2d(channels[3], channels[4], kernel_size=kernel_size,
                             padding=padding, stride=stride),
            PostActivation2d(channels[4], channels[4], kernel_size=kernel_size,
                             padding=padding, stride=stride),
            DefConv(channels[5]),
            PostActivation2d(channels[5], channels[6], kernel_size=kernel_size,
                             padding=padding, stride=stride),
            PostActivation2d(channels[6], channels[6], kernel_size=kernel_size,
                             padding=padding, stride=stride),

        )

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(pool_kernel),
                                Reshape('0', -1),
                                nn.Linear(channels[-1], n_classes)
                                )

    def forward(self, x):
        x = self.net(x)
        return nn.Softmax(dim=1)(self.fc(x))
