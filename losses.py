import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)  # 12, 6, 256, 256
        target = self._one_hot_encoder(target)  # [12, 6, 256, 256]
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
    loss = F.cross_entropy(input=input, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    return loss





# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = cross_entropy(inputs,targets)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

def FocalLoss(ce_loss, alpha=1, gamma=2, reduce=True):
    pt = torch.exp(-ce_loss)
    F_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss_func = cross_entropy
        self.dice_loss_func = DiceLoss(n_classes=2)

    def forward(self, input, target):
        ce_loss = self.ce_loss_func(input, target)
        dice_loss = self.dice_loss_func(input, target, softmax=True)
        return 0.5 * ce_loss + dice_loss

def _one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(2):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def cross_entropy_logits(input, target, weight=None, reduction='mean', softmax=True):
    target = target.long()
    if softmax:
        input = torch.softmax(input, dim=1)
    target = _one_hot_encoder(target)
    weight = torch.zeros_like(target)
    weight = torch.fill_(weight, 0.3)
    weight[target > 0] = 0.7
    # loss = F.cross_entropy(input=input, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, weight=weight, reduction=reduction)
    return loss

class WBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.wce_loss_func = cross_entropy_logits
        self.dice_loss_func = DiceLoss(n_classes=2)
        self.weight = torch.tensor([0.2, 0.8])

    def forward(self, input, target):
        dice_loss = self.dice_loss_func(input, target, softmax=True)
        ce_loss = self.wce_loss_func(input, target, weight=self.weight)
        return 0.5 * ce_loss + dice_loss


class BCEDiceFacolLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss_func = cross_entropy
        self.dice_loss_func = DiceLoss(n_classes=2)
        self.focal_loss_func = FocalLoss

    def forward(self, input, target):
        ce_loss = self.ce_loss_func(input, target)
        dice_loss = self.dice_loss_func(input, target, softmax=True)
        focal_loss = self.focal_loss_func(ce_loss)
        return 0.5 * ce_loss + dice_loss + 0.5 * focal_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


if __name__ == '__main__':
    pre = torch.randn(3, 2, 224, 224)
    target = torch.randn(3, 224, 224)
    Loss = WBCEDiceLoss()
    Loss2 = BCEDiceLoss()
    loss2 = Loss2(pre, target)
    loss = Loss(pre, target)

    # loss = F.cross_entropy(input=pre, target=target, weight=None, reduction='mean')
    print(loss)
    print(loss2)
