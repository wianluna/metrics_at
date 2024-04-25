import torch
from common.data.kadid10k.kadid10k import CustomImageDataset


def add_groups(parser):
    parser.add_argument("--groups", nargs="+", default=["all"], help="kadid-10k groups to run experiments on")
    parser.add_argument("--test_all_groups", action="store_true", help="evaluate model which wasn't adversarily trained")
    return parser


def iterate_dataloaders(opt):
    if opt.test_all_groups:
        opt.groups = ["%02d"%(group) for group in range(1, 26)]
    for group in opt.groups:
        yield (
            torch.utils.data.DataLoader(
                CustomImageDataset(opt.dataroot, group=group),
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            ),
            group
        )
