import torch
import torch.nn as nn
from Exchange_Module import ChannelExchange


class PerturbMask(nn.Module):
    def __init__(self, p):
        super(PerturbMask, self).__init__()
        self.p = p

    def forward(self, x1):
        n, c, h, w = x1.shape
        zero_map = torch.arange(c) % self.p == 0
        map_mask = zero_map.unsqueeze(0).expand((n, -1))
        out_x1 = torch.zeros_like(x1)
        out_x1[~map_mask, ...] = x1[~map_mask, ...]
        return out_x1


class DIM(nn.Module):
    def __init__(self, in_planes):
        super(DIM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.exchange = ChannelExchange()
        self.mask = PerturbMask(4)

    def forward(self, input1, input2):
        input1, input2 = self.exchange(input1, input2)
        diff = torch.sub(input1, input2)
        diff_temp = self.mask(diff)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(diff_temp))))
        ori_out = self.fc2(self.relu1(self.fc1(diff_temp)))
        att = self.sigmoid(avg_out + ori_out)
        feature1 = input1 * att + input1
        feature2 = input2 * att + input2
        difference = torch.sub(feature1, feature2)
        return difference
