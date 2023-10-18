from pathlib import Path
from typing import Any


def get_model_info(model_path: Path, config: dict[str, Any]):
    if config['attack']['train']['type'] == 'none':
        train_method = 'origin'
    elif config['attack']['train']['type'] == 'apgd':
        train_method = 'apgd'
    elif config['attack']['train']['params']['mode'] == 'zero':
        train_method = 'fgsm'
    elif config['attack']['train']['params']['mode'] == 'uniform':
        train_method = 'free_fgsm'

    model_info = {
        'directory': model_path,
        'train_method': train_method,
        'label_strategy': config['label_strategy'],
        'lr_scheduler': config['lr_scheduler']['type'],
    }

    if train_method != 'origin':
        model_info["threat"] = config["attack"]["train"]["params"]["eps"]

    if config['label_strategy'] == 'ssim' or config['label_strategy'] == 'lpips':
        model_info['penalty'] = config['penalty'] if 'penalty' in config else 1

    if 'perc' in config and config['perc']:
        model_info['label_strategy'] += '_perc'

    return model_info
