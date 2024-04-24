import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import os

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        self.reference_dir = os.path.join(dataroot, "images")
        self.compressed_dir = os.path.join(dataroot, "images_compressed")
        transform_list = []
        transform_list = []
        transform_list.append(transforms.Resize(224))
        transform_list += [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.images_list = [img.split('.')[0] for img in os.listdir(self.reference_dir)]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        ref = os.path.join(self.reference_dir, f"{self.images_list[idx]}.png")
        dist = os.path.join(self.compressed_dir, f"{self.images_list[idx]}.jpg")
        ref_img = Image.open(ref).convert("RGB")
        dist_img = Image.open(dist).convert("RGB")
        return self.transform(ref_img), self.transform(dist_img)
