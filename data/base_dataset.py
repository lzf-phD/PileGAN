import torch
import torch.utils.data as data
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import torchvision.transforms as transforms
import numpy as np
import random
import os

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:  #False
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_trans_txt_mc(opt, txt,mask_tensor):
    unique_colors = torch.unique(mask_tensor.permute(1, 2, 0).reshape(-1, 3), dim=0)
    unique_colors = (unique_colors * 255).to(torch.uint8)
    remove_colors = torch.tensor([[255, 255, 255], [132, 132, 132]])
    mask = torch.logical_and(
         (unique_colors != remove_colors[0]).any(dim=1),
         (unique_colors != remove_colors[1]).any(dim=1)
   )
    filtered_colors = unique_colors[mask]
    txt1,txt2,txt3,txt4 = txt[0][::-1],txt[1][::-1],txt[2][::-1],txt[4][::-1]
    if txt4[0]>1.4:#限制最大最小值
        txt4[0] = 1.4
    elif txt4[0]<1.1:
        txt4[0] = 1.1
    # 初始化0张量
    cond1 = torch.zeros((mask_tensor.shape[1], mask_tensor.shape[2]), dtype=torch.float32)
    cond2 = torch.zeros_like(cond1)
    cond3 = torch.zeros_like(cond1)
    cond4 = torch.zeros_like(cond1)
    for i, color in enumerate(filtered_colors):
        mask = (mask_tensor.permute(1, 2, 0) * 255 == color).all(dim=-1)  # (H, W, 3)
        cond4[mask] = 1.0*normalize_parameter(txt4[0],1.1,1.4)  # K
        cond1[mask] = -0.1*normalize_parameter(txt1[i],0,25) #gama
        cond2[mask] = -0.1*normalize_parameter(txt2[i],0,1500) #fa0
        cond3[mask] = -0.9*normalize_parameter(txt3[i],0,350) #qik

    tensor_txt = torch.stack((cond4,cond1, cond2, cond3), dim=0)
    return tensor_txt

def get_trans_txt_dc(opt, txt,mask_tensor):
    unique_colors = torch.unique(mask_tensor.permute(1, 2, 0).reshape(-1, 3), dim=0)
    unique_colors = (unique_colors * 255).to(torch.uint8)
    remove_colors = torch.tensor([[255, 255, 255], [132, 132, 132]])
    mask = torch.logical_and(
         (unique_colors != remove_colors[0]).any(dim=1),
         (unique_colors != remove_colors[1]).any(dim=1)
   )
    filtered_colors = unique_colors[mask]
    txt1,txt2,txt3,txt4 = txt[0][::-1],txt[2][::-1],txt[3][::-1],txt[4][::-1]
    if txt4[0]>2.95:#限制最大最小值
        txt4[0] = 2.95
    elif txt4[0]<2.3:
        txt4[0] = 2.3

    cond1 = torch.zeros((mask_tensor.shape[1], mask_tensor.shape[2]), dtype=torch.float32)
    cond2 = torch.zeros_like(cond1)
    cond3 = torch.zeros_like(cond1)
    cond4 = torch.zeros_like(cond1)
    for i, color in enumerate(filtered_colors):
        mask = (mask_tensor.permute(1, 2, 0) * 255 == color).all(dim=-1)  # (H, W, 3)
        cond4[mask] = 1.0*normalize_parameter(txt4[0],2.3,2.95)  # K
        cond1[mask] = -0.05*normalize_parameter(txt1[i],0,25) #gama
        cond2[mask] = -0.05*normalize_parameter(txt2[i],0,350) #qik
        cond3[mask] = -0.8*normalize_parameter(txt3[i],0,40) #frk

    tensor_txt = torch.stack((cond4, cond1, cond2, cond3), dim=0)
    return tensor_txt

def normalize_parameter(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

