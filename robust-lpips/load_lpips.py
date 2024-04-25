from lpips import LPIPS
import torch


def load_lpips(path):
    net = LPIPS()
    net.load_state_dict(torch.load(path), strict=False)
    return net
