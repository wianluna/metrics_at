from pathlib import Path
import random
from typing import Any

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.transforms.functional import resize, rotate, crop, hflip, to_tensor, normalize

from metric_at.utils import Phase


class IQADataset(Dataset):
    def __init__(
        self,
        dataset: str,
        data_info: str,
        directory: str,
        train_ratio: float,
        train_and_val_ratio: float,
        augment_args: dict[str, Any] | None = None,
        resize_size: tuple[int, int] | None = None,
        normalize: bool = False,
        phase: str = 'train',
    ) -> Dataset:
        self.dataset = dataset
        self.phase = phase
        self.directory = Path(directory)
        self.augment_args = augment_args
        self.normalize = normalize
        self.resize_size = resize_size

        info = h5py.File(data_info, 'r')
        index = info['index'][:, 0]

        ref_ids = info['ref_ids'][0, :]
        index_len = len(index)
        if phase == 'train':
            index = index[: int(train_ratio * index_len)]
        elif phase == 'val':
            index = index[int(train_ratio * index_len) : int(train_and_val_ratio * index_len)]
        elif phase == 'test':
            index = index[int(train_and_val_ratio * index_len) :]

        self.index = []
        for i, id_i in enumerate(ref_ids):
            if id_i in index:
                self.index.append(i)
        print(f'# {phase} images: {len(self.index)}')

        self.label = info['subjective_scores'][0, self.index].astype(np.float32)
        self.label_std = info['subjective_scoresSTD'][0, self.index].astype(np.float32)

        self.image_names = [
            info[info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index
        ]
        self.images = []
        for image_name in self.image_names:
            image = Image.open(self.directory / image_name).convert('RGB')
            if dataset == 'CLIVE':
                w, h = image.size
                if w != 500 or h != 500:
                    image = resize(image, (500, 500))
            if resize_size:
                image = resize(image, resize_size)
            self.images.append(image)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # image = Image.open(self.directory / self.image_names[idx]).convert('RGB')
        # if self.dataset == 'CLIVE':
        #     w, h = image.size
        #     if w != 500 or h != 500:
        #         image = resize(image, (500, 500))
        # if self.resize_size:
        #     image = resize(image, self.resize_size)

        image = self.transform(self.images[idx])
        label = self.label[idx]
        label_std = self.label_std[idx]
        return image, (label, label_std)

    def transform(self, image: Image.Image):
        if self.phase == 'train' and self.augment_args:
            angle = random.uniform(-self.augment_args['angle'], self.augment_args['angle'])
            image = rotate(image, angle)
            if random.random() < self.augment_args['hflip_p']:
                image = hflip(image)

            w, h = image.size
            i = random.randint(0, h - self.augment_args['crop_size'])
            j = random.randint(0, w - self.augment_args['crop_size'])
            image = crop(
                image, i, j, self.augment_args['crop_size'], self.augment_args['crop_size']
            )

        image = to_tensor(image)
        if self.normalize:
            image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image


def get_data_loaders(
    rank: int,
    num_tasks: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    args: dict[str, Any],
    phase: Phase = Phase.TRAIN,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare the train-val-test data.

    :param args: related arguments
    :return: train_loader, val_loader, test_loader
    """
    if phase == Phase.TRAIN:
        train_set = IQADataset(**args, phase='train')
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=num_tasks,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if rank:
            return train_loader, None, None

        val_set = IQADataset(**args, phase='val')
        val_loader = DataLoader(val_set, batch_size=batch_size)

    test_set = IQADataset(**args, phase='test')
    test_loader = DataLoader(test_set, batch_size=batch_size)
    if phase == Phase.TEST:
        return test_loader

    return train_loader, val_loader, test_loader
