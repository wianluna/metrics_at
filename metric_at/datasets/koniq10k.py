from pathlib import Path
import random
from typing import Any

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.transforms.functional import resize, rotate, crop, hflip, to_tensor, normalize

from metric_at.utils import Phase


class Koniq10kDataset(Dataset):
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

        self.data_path = self.directory / phase / '1024x768'
        print(self.data_path)
        self.data_info = pd.read_csv(self.data_path / 'metadata.csv')

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        image_info = self.data_info.iloc[idx]
        image = Image.open(self.data_path / image_info['image']).convert('RGB')
        # if self.dataset == 'CLIVE':
        #     w, h = image.size
        #     if w != 500 or h != 500:
        #         image = resize(image, (500, 500))
        if self.resize_size:
            image = resize(image, self.resize_size)

        image = self.transform(image)
        label = image_info['label']
        label_std = image_info['label_std']
        percentile = image_info['percentile']
        return image, (label, label_std, percentile)

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
        train_set = Koniq10kDataset(**args, phase='train')
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

        val_set = Koniq10kDataset(**args, phase='val')
        val_loader = DataLoader(val_set, batch_size=batch_size)

    test_set = Koniq10kDataset(**args, phase='test')
    test_loader = DataLoader(test_set, batch_size=batch_size)
    if phase == Phase.TEST:
        return test_loader

    return train_loader, val_loader, test_loader
