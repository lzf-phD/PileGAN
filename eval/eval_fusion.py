import os
import statistics

import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

# path
result_path = '../results/PileGAN/test_latest/images'

file_input = [f for f in os.listdir(result_path) if f.endswith('_input_label.jpg')]
file_real = [f for f in os.listdir(result_path) if f.endswith('_real_image.jpg')]
file_syn = [f for f in os.listdir(result_path) if f.endswith('_synthesized_image.jpg')]

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


def get_nearest_color_indices(image_tensor, fusion_colors):
    H, W = image_tensor.shape[1:]
    image_flat = (image_tensor * 255).permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)

    distances = torch.cdist(image_flat.unsqueeze(0), fusion_colors.unsqueeze(0)).squeeze(0)  # (H*W, 10)

    nearest_indices = torch.argmin(distances, dim=1)  # (H*W,)

    return nearest_indices.reshape(H, W)  # (H, W)

list_red = []
list_blue = []
for i in range(len(file_input)):
    input_path = os.path.join(result_path, file_input[i])
    real_path = os.path.join(result_path, file_real[i])
    fake_path = os.path.join(result_path, file_syn[i])
    imageA = Image.open(input_path).convert("RGB")
    imageB = Image.open(real_path).convert("RGB")
    imageC = Image.open(fake_path).convert("RGB")
    # tensor(C, H, W)
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
    imageB_change = image_list[0]
    imageC_change = image_list[1]

    real_indices = get_nearest_color_indices(imageB_change, fusion_colors)
    fake_indices = get_nearest_color_indices(imageC_change, fusion_colors)

    # fusion matrix
    confusion_matrix = torch.zeros(9, 9, dtype=torch.int32)

    for i in range(real_indices.shape[0]):
        for j in range(real_indices.shape[1]):
            real_idx = real_indices[i, j].item()
            fake_idx = fake_indices[i, j].item()
            confusion_matrix[real_idx, fake_idx] += 1
    PA = confusion_matrix.diagonal().sum().item()/524288

    if confusion_matrix[0, 0]>0:
       list_red.append(PA)
    else:
       list_blue.append(PA)

    total_pixels = confusion_matrix.sum().item()
    #print("Total pixels in confusion matrix:", total_pixels)

# mean
mean_red = statistics.mean(list_red)
mean_blue = statistics.mean(list_blue)

# std
stdev_red = statistics.stdev(list_red)
stdev_blue = statistics.stdev(list_blue)

print(len(list_red))
print(len(list_blue))
print(mean_red)
print(mean_blue)
print(stdev_red)
print(stdev_blue)
