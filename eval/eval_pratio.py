import os
from collections import Counter

import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import statistics

result_path = '../results/PileGAN/test_latest/images'

file_input = [f for f in os.listdir(result_path) if f.endswith('_input_label.jpg')]
file_real = [f for f in os.listdir(result_path) if f.endswith('_real_image.jpg')]
file_syn = [f for f in os.listdir(result_path) if f.endswith('_synthesized_image.jpg')]


# 定义要去掉的颜色
remove_colors = torch.tensor([[255, 255, 255], [132, 132, 132]])
target_colors = torch.tensor([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 205, 0],
    [0, 155, 0], [0, 105, 0], [0, 55, 0], [255, 255, 255],
    [132, 132, 132]
], dtype=torch.float32)

fusion_colors = torch.tensor([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 205, 0],
    [0, 155, 0], [0, 105, 0], [0, 55, 0], [255, 255, 255],
    [132, 132, 132]
], dtype=torch.float32)

name = []
list = []
list_syn = []
list_width = []
for i in range(len(file_input)):
    input_path = os.path.join(result_path, file_input[i])
    real_path = os.path.join(result_path, file_real[i])
    fake_path = os.path.join(result_path, file_syn[i])
    imageA = Image.open(input_path).convert("RGB")
    imageB = Image.open(real_path).convert("RGB")
    imageC = Image.open(fake_path).convert("RGB")

    transform = transforms.ToTensor()
    imageA_tensor = transform(imageA)  # (3, H, W)
    imageB_tensor = transform(imageB)  # (3, H, W)
    imageC_tensor = transform(imageC)  # (3, H, W)

    image_list = [imageB_tensor, imageC_tensor]

    for i in range(len(image_list)):
        image_tensor = image_list[i]
        image_tensor = (image_tensor * 255).byte().permute(1, 2, 0).float()

        H = 1024
        W = 512
        image_flat = image_tensor.reshape(-1, 3)

        distances = torch.cdist(image_flat.unsqueeze(0), target_colors.unsqueeze(0)).squeeze(0)

        nearest_indices = torch.argmin(distances, dim=1)

        image_change = target_colors[nearest_indices]

        image_change = image_change.reshape(H, W, 3).byte()
        image_change = image_change.permute(2, 0, 1).float() / 255.0
        image_list[i] = image_change

    target_gray = torch.tensor([132, 132, 132], dtype=torch.float32).view(3, 1, 1)
    mask_A = (torch.abs(imageA_tensor * 255 - target_gray) < 1e-3).all(dim=0)


    red_color = torch.tensor([255, 0, 0], dtype=torch.float32).view(3, 1, 1)
    blue_color = torch.tensor([0, 0, 255], dtype=torch.float32).view(3, 1, 1)


    imageC_selected = image_list[1][:, mask_A]
    imageB_selected = image_list[0][:, mask_A]

    count_red_C = (torch.abs(imageC_selected * 255 - red_color.view(3, 1)) < 1e-3).all(dim=0).sum().item()
    count_blue_C = (torch.abs(imageC_selected * 255 - blue_color.view(3, 1)) < 1e-3).all(dim=0).sum().item()

    count_red_B = (torch.abs(imageB_selected * 255 - red_color.view(3, 1)) < 1e-3).all(dim=0).sum().item()
    count_blue_B = (torch.abs(imageB_selected * 255 - blue_color.view(3, 1)) < 1e-3).all(dim=0).sum().item()

    a = (count_red_C + count_blue_C) / (count_red_B + count_blue_B)

    list.append(a)

mean = statistics.mean(list)
stdev = statistics.stdev(list)

print(list)
print(mean)
print(stdev)

