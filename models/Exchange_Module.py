import torch
from mmengine.registry import Registry
from mmengine.model import BaseModule

INTERACTION_LAYERS = Registry('interaction layer')


@INTERACTION_LAYERS.register_module()
class ChannelExchange(BaseModule):
    def __init__(self, p=1 / 2):
        super(ChannelExchange, self).__init__()
        assert 0 <= p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        n, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((n, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


@INTERACTION_LAYERS.register_module()
class SpatialExchange(BaseModule):
    def __init__(self, p=1 / 2):
        super(SpatialExchange, self).__init__()
        assert 0 <= p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        n, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2
