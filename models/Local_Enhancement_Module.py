import math
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, groups=1, width_per_group=64):
        super(ConvBlock, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out


class LFE(nn.Module):
    def __init__(self, in_channel, ratio=4, b=1, gamma=2):
        super(LFE, self).__init__()
        self.conv = ConvBlock(in_channel, in_channel)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.mlp1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.mlp2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        b, c, h, w = inputs.shape
        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)
        x_max_pool = self.relu(self.mlp1(max_pool.view([b, c])))
        x_avg_pool = self.relu(self.mlp1(avg_pool.view([b, c])))
        x_pool = x_max_pool + x_avg_pool
        x_pool = self.mlp2(x_pool).view([b, 1, c])
        x_pool = self.sigmoid(self.conv1d(x_pool)).view([b, c, 1, 1])
        outputs = inputs * x_pool
        return outputs
