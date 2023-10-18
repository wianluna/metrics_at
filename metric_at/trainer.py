from datetime import datetime
import os
from pathlib import Path
import re
import time
import shutil
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pyiqa
from pytorch_msssim import ms_ssim, ssim
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from metric_at.attacks.fgsm import FGSM
from metric_at.attacks.pgd import PGD, AutoPGD
from metric_at.attacks.base import Attacker
from metric_at.model import IQAModel, normalize_model
from metric_at.datasets.koniq10k import get_data_loaders
from metric_at.log import dump_config
from metric_at.loss import IQALoss
from metric_at.lr_scheduler import CosineLRScheduler, InterpolationLRScheduler
from metric_at.metrics import IQAPerformance, dump_scalar_metrics
from metric_at.utils import load_config, init_torch_seeds, Phase
from metric_at.model_info import get_model_info


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['KMP_WARNINGS'] = 'off'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

MAX_SCORE = 88.3888888889
MIN_SCORE = 3.91176470588
MAX_SCORE_NORM = 1.0982177
MIN_SCORE_NORM = 0.29622972


class AdversarialTrainer:
    def __init__(self, gpu: int, config_path: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.eval_only = self.config['eval_only']

        self.world_size = torch.cuda.device_count()
        self.distributed = self.world_size > 1
        self.gpu = gpu
        print(f'#{self.gpu}')

        init_torch_seeds(self.config['seed'])

        if self.distributed:
            self.setup_distributed()

        if not self.config['data']['normalize']:
            self.model = normalize_model(IQAModel(**self.config['model']))
        else:
            self.model = normalize_model(
                IQAModel(**self.config['model']),
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            )
        self.model.to(self.gpu)

        checkpoint = torch.load('/home/a.chistyakova/LinearityIQA/checkpoints/my_p1q2.pth')
        self.model.model.load_state_dict(checkpoint['model'])

        self.k = [1, 1, 1]
        self.b = [0, 0, 0]

        self._init_logger()

    def __del__(self):
        if self.gpu == 0:
            self.writer.close()

    def _init_logger(self):
        if self.gpu == 0:
            if self.config['checkpoint_path']:
                self.log_dir = Path(self.config['checkpoint_path'])
                self.writer = SummaryWriter(log_dir=self.log_dir)
                return

            if self.config['attack']['train']['type'] == 'none':
                train_method = 'origin'
            elif self.config['attack']['train']['type'] == 'apgd':
                train_method = 'apgd'
            elif self.config['attack']['train']['params']['mode'] == 'zero':
                train_method = 'fgsm'
            elif self.config['attack']['train']['params']['mode'] == 'uniform':
                train_method = 'free_fgsm'
            else:
                raise NotImplementedError

            threat = self.config["attack"]["train"]["params"]["eps"]
            exp = f'ep={self.config["train"]["epochs"]}_eps={threat}'

            self.log_dir = (
                Path(self.config['log']['directory']).resolve()
                / train_method
                / f"{self.config['label_strategy']}"
                f"{('_' + self.config['penalty']) if 'penalty' in self.config else ''}"
                / f'{str(datetime.now())[:-4]}_{exp}_{self.config["lr_scheduler"]["type"]}'
            )
            self.log_dir.mkdir(parents=True, exist_ok=True)

            self.writer = SummaryWriter(log_dir=self.log_dir)

            print(f'=> Logging in {self.log_dir}')
            dump_config(self.config, self.writer)

            # try not to lose the best presets
            shutil.copy(self.config_path, self.log_dir / 'presets.yaml')

            self.start_training_time = time.time()

    def train(self):
        self._prepare_for_training()
        self._train_loop()

    def _prepare_for_training(self):
        self.current_epoch = 0
        self.end_epoch = self.config['train']['epochs']

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            rank=self.gpu,
            num_tasks=self.world_size,
            args=self.config['data'],
            batch_size=self.config['train']['batch_size'],
            num_workers=self.config['train']['num_workers'],
            seed=self.config['seed'],
        )

        self._init_optimizer(**self.config['optimizer'])
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu])
        self.scaler = GradScaler()

        self._init_lr_scheduler()
        self.loss = IQALoss(**self.config['loss'])
        self.train_attack = self._init_attack(self.config['attack']['train'], Phase.TRAIN)

        if self.gpu == 0:
            self.val_criterion = self.config["train"]["val_criterion"]
            self.metric_computer = IQAPerformance(
                phase=Phase.VAL, k=[1, 1, 1], b=[0, 0, 0], mapping=True
            )
            self.best_val_criterion, self.best_epoch = -100, -1

    def _init_optimizer(
        self, learning_rate: float, ft_lr_ratio: float, weight_decay: float
    ) -> None:
        self.optimizer = Adam(
            [
                {'params': self.model.model.regression.parameters()},
                {'params': self.model.model.dr6.parameters()},
                {'params': self.model.model.dr7.parameters()},
                {'params': self.model.model.regr6.parameters()},
                {'params': self.model.model.regr7.parameters()},
                {
                    'params': self.model.model.features.parameters(),
                    'lr': learning_rate * ft_lr_ratio,
                },
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _init_lr_scheduler(self) -> None:
        if self.config['lr_scheduler']['type'] == 'cosine':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                num_epochs=self.config['train']['epochs'],
                lr_peak=self.config['lr_scheduler']['lr_peak'],
                lr_peak_epoch=self.config['lr_scheduler']['lr_peak_epoch'],
                num_steps_per_epoch=len(self.train_loader),
            )
        elif self.config['lr_scheduler']['type'] == 'interp':
            self.lr_scheduler = InterpolationLRScheduler(
                self.optimizer,
                self.config['lr_scheduler']['points'],
                num_epochs=self.config['train']['epochs'],
                num_steps_per_epoch=len(self.train_loader),
            )
        elif self.config['lr_scheduler']['type'] == 'step':
            lr_decay_step = int(
                self.end_epoch
                / (
                    1
                    + np.log(self.config['lr_scheduler']['overall_lr_decay'])
                    / np.log(self.config['lr_scheduler']['lr_decay'])
                )
            )
            self.lr_scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_decay_step,
                gamma=self.config['lr_scheduler']['lr_decay'],
            )
        elif self.config['lr_scheduler']['type'] == 'cyclic':
            self.lr_scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=5e-6,
                max_lr=self.config['lr_scheduler']['lr_peak'],
                step_size_up=self.config['lr_scheduler']['step_size_up'] * len(self.train_loader),
                cycle_momentum=False,
            )

    def _init_attack(
        self,
        attack_config: dict[str, Any],
        phase: Phase,
        metric_range: tuple[float, float] | None = None,
    ) -> Attacker:
        attack_name = attack_config['type']
        if attack_name == 'none':
            return None

        attackers = {'fgsm': FGSM, 'pgd': PGD, 'apgd': AutoPGD}
        attacker_cls = attackers.get(attack_name)

        if attacker_cls is None:
            raise RuntimeError(f'Unknown attack `{attack_name}`')

        if metric_range:

            def loss_computer(y, target):
                return -torch.sum(1 - (y[-1] * self.k[0] + self.b[0]) / metric_range)

        else:
            loss_computer = self.loss

        attacker = attacker_cls(
            model=self.model,
            loss_computer=loss_computer,
            **attack_config['params'],
        )

        return attacker

    def _train_loop(self):
        train_data_len = len(self.train_loader)

        if self.config['label_strategy'] == 'lpips':
            self.proxy_metric = pyiqa.create_metric(
                'lpips', as_loss=False, net='vgg', device=self.gpu
            )

        while self.current_epoch < self.end_epoch:
            self.model.train()

            done_steps = self.current_epoch * train_data_len
            batch_start_time = time.time()

            for step, (inputs, label) in enumerate(self.train_loader):
                metrics = self._train_step(inputs, label, step, batch_start_time)
                if self.gpu == 0:
                    dump_scalar_metrics(
                        metrics, self.writer, Phase.TRAIN, global_step=done_steps + step
                    )

                batch_start_time = time.time()

            if self.config['lr_scheduler']['type'] == 'step':
                self.lr_scheduler.step()
            self.current_epoch += 1

            if self.gpu:
                continue

            self.metric_computer.reset()
            self.model.eval()
            for step, (inputs, label) in enumerate(self.val_loader):
                self._val_step(inputs, label)

            metrics = self.metric_computer.compute()
            dump_scalar_metrics(
                metrics,
                self.writer,
                Phase.VAL,
                global_step=self.current_epoch,
                dataset=self.config['data']['dataset'],
            )

            val_criterion = abs(metrics[self.val_criterion])
            if val_criterion >= self.best_val_criterion:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'k': self.k,
                    'b': self.b,
                    'epoch': self.current_epoch,
                }
                torch.save(checkpoint, self.log_dir / 'best_model.pth')

                self.best_val_criterion = val_criterion
                self.best_epoch = self.current_epoch
                print(
                    f'Save current best model @best_val_criterion ({self.val_criterion}):\
                          {self.best_val_criterion:.3f} @epoch: {self.best_epoch}'
                )
            else:
                print(
                    f'Model is not updated @val_criterion ({self.val_criterion}):\
                          {val_criterion:.3f} @epoch: {self.current_epoch}'
                )

        if self.gpu:
            return

        self.metric_computer = IQAPerformance(
            phase=Phase.TRAIN, k=[1, 1, 1], b=[0, 0, 0], mapping=True
        )

        # save model on the last step
        self.metric_computer.reset()
        for step, (inputs, label) in enumerate(self.train_loader):
            self._val_step(inputs, label)
        coeffs = self.metric_computer.compute()
        preds = self.metric_computer.preds
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'k': coeffs['k'],
            'b': coeffs['b'],
            'min': preds.min(),
            'max': preds.max(),
            'epoch': self.current_epoch,
        }
        torch.save(checkpoint, self.log_dir / 'final_model.pth')

        # Update coefficients for the best model
        checkpoint = torch.load(self.log_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.metric_computer.reset()
        for step, (inputs, label) in enumerate(self.train_loader):
            self._val_step(inputs, label)

        coeffs = self.metric_computer.compute()
        print(coeffs)
        checkpoint['k'] = coeffs['k']
        checkpoint['b'] = coeffs['b']
        preds = self.metric_computer.preds
        checkpoint['max'] = preds.max()
        checkpoint['min'] = preds.min()
        torch.save(checkpoint, self.log_dir / 'best_model.pth')

    def _train_step(self, inputs: torch.Tensor, label: torch.Tensor, step: int, start_time: float):
        metrics = {}
        inputs = inputs.cuda(self.gpu, non_blocking=True)
        label = [k.cuda(self.gpu, non_blocking=True) for k in label]
        metrics['data_time'] = time.time() - start_time

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            if self.train_attack:
                adv_inputs = self.train_attack.run(inputs, label)

                if self.config['label_strategy'] == 'min':
                    # set min label for adversarial samples
                    label_shape = label[0].shape
                    adv_label = (
                        torch.full(label_shape, MIN_SCORE),
                        torch.full(label_shape, MIN_SCORE_NORM),
                    )
                    adv_label = [k.cuda(self.gpu, non_blocking=True) for k in adv_label]
                    inputs = torch.concat([adv_inputs, inputs])
                    label = [torch.concat([adv_lab, lab]) for adv_lab, lab in zip(adv_label, label)]
                elif self.config['label_strategy'] == 'ssim':
                    # adjust label via SSIM
                    if 'penalty' not in self.config:
                        self.config['penalty'] = 1
                    ssim_val = ssim(adv_inputs, inputs, data_range=1, size_average=False)
                    adv_label = [
                        torch.clamp(
                            label[0].clone()
                            - self.config['penalty'] * (1 - ssim_val) * (MAX_SCORE - MIN_SCORE),
                            MIN_SCORE,
                            MAX_SCORE,
                        ),
                        torch.clamp(
                            label[1].clone()
                            - self.config['penalty']
                            * (1 - ssim_val)
                            * (MAX_SCORE_NORM - MIN_SCORE_NORM),
                            MIN_SCORE_NORM,
                            MAX_SCORE_NORM,
                        ),
                    ]

                    inputs = torch.concat([adv_inputs, inputs])
                    label = [torch.concat([adv_lab, lab]) for adv_lab, lab in zip(adv_label, label)]
                elif self.config['label_strategy'] == 'ms_ssim':
                    # adjust label via MS-SSIM
                    if 'penalty' not in self.config:
                        self.config['penalty'] = 1

                    ms_ssim_val = ms_ssim(adv_inputs, inputs, data_range=1, size_average=False)
                    adv_label = [
                        torch.clamp(
                            label[0].clone()
                            - self.config['penalty'] * (1 - ms_ssim_val) * (MAX_SCORE - MIN_SCORE),
                            MIN_SCORE,
                            MAX_SCORE,
                        ),
                        torch.clamp(
                            label[1].clone()
                            - self.config['penalty']
                            * (1 - ms_ssim_val)
                            * (MAX_SCORE_NORM - MIN_SCORE_NORM),
                            MIN_SCORE_NORM,
                            MAX_SCORE_NORM,
                        ),
                    ]

                    inputs = torch.concat([adv_inputs, inputs])
                    label = [torch.concat([adv_lab, lab]) for adv_lab, lab in zip(adv_label, label)]
                elif self.config['label_strategy'] == 'lpips':
                    # adjust label via LPIPS
                    if 'penalty' not in self.config:
                        self.config['penalty'] = 1

                    lpips = self.proxy_metric(adv_inputs, inputs)
                    adv_label = [
                        torch.clamp(
                            label[0].clone()
                            - self.config['penalty'] * lpips * (MAX_SCORE - MIN_SCORE),
                            MIN_SCORE,
                            MAX_SCORE,
                        ),
                        torch.clamp(
                            label[1].clone()
                            - self.config['penalty'] * lpips * (MAX_SCORE_NORM - MIN_SCORE_NORM),
                            MIN_SCORE_NORM,
                            MAX_SCORE_NORM,
                        ),
                    ]

                    inputs = torch.concat([adv_inputs, inputs])
                    label = [torch.concat([adv_lab, lab]) for adv_lab, lab in zip(adv_label, label)]
                elif bool(re.search(r'\d', str(self.config['label_strategy']))):
                    # -5% / -10%
                    # label[1] is std label and it isn't correct subtract 5% in this way,
                    # but it is unused in training and here for uniformity
                    penalty = float(self.config['label_strategy']) / 100
                    adv_label = [
                        torch.clamp(
                            label[0].clone() - penalty * (MAX_SCORE - MIN_SCORE),
                            MIN_SCORE,
                            MAX_SCORE,
                        ),
                        torch.clamp(
                            label[1].clone() - penalty * (MAX_SCORE_NORM - MIN_SCORE_NORM),
                            MIN_SCORE_NORM,
                            MAX_SCORE_NORM,
                        ),
                    ]

                    adv_label = [k.cuda(self.gpu, non_blocking=True) for k in adv_label]
                    inputs = torch.concat([adv_inputs, inputs])
                    label = [torch.concat([adv_lab, lab]) for adv_lab, lab in zip(adv_label, label)]
                else:
                    inputs = adv_inputs

            model_out = self.model(inputs)
            loss = self.loss(model_out, label)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.config['lr_scheduler']['type'] == 'step':
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
        elif self.config['lr_scheduler']['type'] == 'cyclic':
            self.lr_scheduler.step()
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
        else:
            metrics['lr'] = self.lr_scheduler.adjust_learning_rate(
                self.current_epoch,
                step=step,
            )
        metrics['total_loss'] = loss.cpu().detach().numpy()
        metrics['total_time'] = time.time() - start_time
        return metrics

    def _val_step(self, inputs: torch.Tensor, label: torch.Tensor, attack: Attacker = None):
        inputs = inputs.cuda(self.gpu, non_blocking=True)
        label = [k.cuda(self.gpu, non_blocking=True) for k in label]

        if attack:
            inputs = attack.run(inputs, label)

        model_out = self.model(inputs)
        self.metric_computer.update(model_out, label)

    def eval(self):
        checkpoint = torch.load(self.log_dir / 'best_model.pth')
        results = {}
        self.model.load_state_dict(checkpoint['model'])
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.metric_computer = IQAPerformance(phase=Phase.TEST, k=self.k, b=self.b, mapping=True)
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric range: {metric_range}')

        # log model
        checkpoint = torch.load(self.log_dir / 'best_model.pth')
        model_info = get_model_info(self.log_dir, self.config)

        if self.config['data']['dataset'] == 'NIPS2017':
            from metric_at.datasets.nips import NipsDataset

            dataset = NipsDataset(
                rank=self.gpu,
                num_tasks=self.world_size,
                seed=self.config['seed'],
                **self.config['data'],
            )
            self.test_loader = dataset.get_test_loader()
        else:
            self.test_loader = get_data_loaders(
                rank=self.gpu,
                num_tasks=self.world_size,
                args=self.config['data'],
                batch_size=self.config['train']['batch_size'],
                num_workers=self.config['train']['num_workers'],
                seed=self.config['seed'],
                phase=Phase.TEST,
            )

        self.model.eval()
        self.metric_computer.reset()
        for step, (inputs, label) in enumerate(self.test_loader):
            self._val_step(inputs, label)
        metrics = self.metric_computer.compute()
        print(metrics)
        model_info['SRCC'] = metrics['SROCC']

        orig_preds = self.metric_computer.preds.copy()
        orig_preds_scaled = (orig_preds - checkpoint['min']) / metric_range
        results['orig_preds'] = orig_preds

        for attack_args in self.config['attack']['test']:
            attack = self._init_attack(attack_args, Phase.TEST, metric_range=metric_range)

            self.metric_computer.reset()
            for step, (inputs, label) in enumerate(self.test_loader):
                self._val_step(inputs, label, attack)

            self.metric_computer.compute()
            att_preds = self.metric_computer.preds
            att_preds_scaled = (att_preds - checkpoint['min']) / metric_range
            results[f'{attack_args["params"]["eps"]}/255'] = orig_preds

            abs_gain = np.mean(att_preds - orig_preds)
            print(f'Abs gain for eps={attack_args["params"]["eps"]}: {abs_gain}')

            abs_gain = np.mean(att_preds_scaled - orig_preds_scaled)
            print(f'Abs gain scaled for eps={attack_args["params"]["eps"]}: {abs_gain}')

        pd.DataFrame.from_dict(results).to_csv(
            self.log_dir / f'results_{self.config["data"]["dataset"]}.csv', index=False
        )

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def setup_distributed(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        dist.init_process_group('nccl', rank=self.gpu, world_size=self.world_size)
        torch.cuda.set_device(self.gpu)

    @classmethod
    def run(cls, *args):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            multiprocessing.spawn(cls._exec_wrapper, args=args, nprocs=world_size, join=True)
        else:
            cls.exec(0, *args)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, config_path):
        trainer = cls(gpu=gpu, config_path=config_path)
        if trainer.eval_only:
            trainer.eval()
        else:
            trainer.train()
            if gpu == 0:
                trainer.eval()
        if trainer.distributed:
            trainer.cleanup_distributed()
