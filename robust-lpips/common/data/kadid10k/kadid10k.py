import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import os

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        annotations_file = os.path.join(dataroot, "dmos.csv")
        self.mos = pd.read_csv(annotations_file)
        self.img_dir = os.path.join(dataroot, "images")
        transform_list = []
        transform_list.append(transforms.Resize(224))
        transform_list += [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        ref = os.path.join(self.img_dir, self.mos.iloc[idx, 0])
        dist = os.path.join(self.img_dir, self.mos.iloc[idx, 1])
        mos = self.mos.iloc[idx, 2]
        ref_img = Image.open(ref)
        dist_img = Image.open(dist)
        return self.transform(ref_img), self.transform(dist_img), mos
