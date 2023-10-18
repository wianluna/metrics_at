from pathlib import Path

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


class NipsDataset(Dataset):
    def __init__(
        self,
        rank: int,
        num_tasks: int,
        seed: int,
        dataset: str,
        directory: str,
        batch_size: int,
        num_workers: int,
        normalize: bool,
        resize_size: tuple[int, int] | None = None,
    ) -> Dataset:
        self.directory = Path(directory)
        if resize_size:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        self.rank = rank
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def get_test_set(self) -> Dataset:
        return ImageFolder(
            root=self.directory,
            transform=self.transform,
        )

    def get_test_loader(self):
        test_set = self.get_test_set()

        test_sampler = DistributedSampler(
            test_set, num_replicas=self.num_tasks, rank=self.rank, shuffle=False
        )

        return DataLoader(
            test_set,
            sampler=test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
