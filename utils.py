import argparse
import copy
import os

import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import utils
import numpy as np
import torch


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def visualize_pred(pre):
    pred = torch.argmax(pre, dim=1, keepdim=True)
    pred_vis = pred * 255
    return pred_vis


def draw(batch, pre, save_path, num, x, split):
    vis_input = make_numpy_grid(de_norm(batch['A']))
    vis_input2 = make_numpy_grid(de_norm(batch['B']))
    vis_gt = make_numpy_grid(batch['L'])
    vis_pred = make_numpy_grid(visualize_pred(pre))
    vis = np.concatenate([vis_input, vis_input2, vis_gt, vis_pred], axis=0)
    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(
        save_path, f"{split}_batch_{num}_{x}.png")
    plt.imsave(file_name, vis)


def draw_output(pre, save_path, name):
    vis_pred = make_numpy_grid(visualize_pred(pre))
    vis = np.concatenate([vis_pred], axis=0)
    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    file_name = os.path.join(save_path, name[0])
    plt.imsave(file_name, vis)


def load_from(pretrained_path, model):
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = model.load_state_dict(pretrained_dict, strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = model.state_dict()
        full_dict = copy.deepcopy(model_dict)
        with open("model.txt", "w", encoding="utf-8") as fp:
            for k, v in model_dict.items():
                fp.write(f"{k}\n")
        with open("pretrain.txt", "w", encoding="utf-8") as fp:
            for k, v in pretrained_dict.items():
                fp.write(f"{k}\n")
        model_keys = model_dict.keys()
        # for i in model_keys:
        #     print(i)
        for k, v in pretrained_dict.items():
            for i in range(1, 4):
                if i != 3:
                    k_new = "transformer_branch." + k.replace("encoder_layers", f"encoder_layers{i}")
                    if k_new in model_keys:
                        full_dict.update({k_new: v})
                        # print(f"{k_new} has updated!!")
                else:
                    k_new = "transformer_branch." + k.replace("encoder_layers.3", "Fusion_layer")
                    if k_new in model_keys:
                        full_dict.update({k_new: v})
                        # print(f"{k_new} has updated!!")

        # for k, v in pretrained_dict.items():
        #     if "layers1." in k:
        #         current_layer_num = 3 - int(k[7:8])
        #         current_k = "layers_up." + str(current_layer_num) + k[8:]
        #         full_dict.update({current_k: v})
        # for k in list(full_dict.keys()):
        #     if k in model_dict:
        #         if full_dict[k].shape != model_dict[k].shape:
        #             print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
        #             del full_dict[k]

        msg = model.load_state_dict(full_dict, strict=False)
        print(msg)
    else:
        print("none pretrain")


def Get_dataset_path(dataset_name):
    if dataset_name == "LEVIR" or dataset_name == "LEVIR-CD":
        return "/data/lmt/Datasets/LEVIR_Dataset/LEVIR_224_overlap"
    if dataset_name == "WHU-CD":
        return "/data/lmt/Datasets/WHU-CD_dataset/WHU-CD-224-overlap"
