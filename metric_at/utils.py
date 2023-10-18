from enum import Enum
from pathlib import Path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3
    TEST_BEST = 4


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)


# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)


def load_config(config_path: Path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as exc:
        raise RuntimeError(f'Cannot load configuration {config_path}') from exc

    return config


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 1):
    """
    Set the seed for generating random numbers.

    :param seed: the desired seed
    """
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
