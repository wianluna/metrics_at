import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import os

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, group):
        annotations_file = os.path.join(dataroot, "dmos.csv")
        self.img_dir = os.path.join(dataroot, "images")
        self.data = []
        for _, entry in pd.read_csv(annotations_file).iterrows():
            ref = entry.ref_img
            dist = entry.dist_img
            mos = entry.dmos
            if group == "all" or dist.split('_')[1] == group:
                self.data.append((ref, dist, mos))

        transform_list = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ref = os.path.join(self.img_dir, self.data[idx][0])
        dist = os.path.join(self.img_dir, self.data[idx][1])
        mos = self.data[idx][2]

        ref_img = Image.open(ref)
        dist_img = Image.open(dist)

        return self.transform(ref_img), self.transform(dist_img), mos
