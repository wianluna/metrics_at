from typing import Dict

import numpy as np
from scipy import stats
from pyiqa.models.inference_model import InferenceModel
import torch
from torch.utils.tensorboard import SummaryWriter

from metric_at.utils import Phase


class IQAPerformance:
    """
    Evaluation of VQA methods using SROCC, PLCC, RMSE.
    """

    def __init__(
        self,
        phase: Phase = Phase.TRAIN,
        k: list[float] = [1, 1, 1],
        b: list[float] = [0, 0, 0],
        mapping: bool = True,
    ):
        self.k = k
        self.b = b
        self.phase = phase
        self.mapping = mapping

    def reset(self):
        self._y_pred = []
        self._y_pred1 = []
        self._y_pred2 = []
        self._y = []
        self._y_std = []

    def update(self, y_pred, y):
        self._y.extend([t.item() for t in y[0]])
        self._y_std.extend([t.item() for t in y[1]])
        self._y_pred.extend([t.item() for t in y_pred[-1]])
        self._y_pred1.extend([t.item() for t in y_pred[0]])
        self._y_pred2.extend([t.item() for t in y_pred[1]])

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))

        pq_before = np.reshape(np.asarray(self._y_pred), (-1, 1))
        self.preds = self.linear_mapping(pq_before, sq, i=0)

        SROCC = stats.spearmanr(sq, self.preds)[0]
        PLCC = stats.pearsonr(sq, self.preds)[0]
        RMSE = np.sqrt(((sq - self.preds) ** 2).mean())

        if self.phase == Phase.TRAIN:
            return {'k': self.k, 'b': self.b}
        return {
            'SROCC': SROCC,
            'PLCC': PLCC,
            'RMSE': RMSE,
        }

    def get_preds(self):
        self.compute()
        return self.preds.copy()

    def linear_mapping(self, pq, sq, i=0):
        if not self.mapping:
            return np.reshape(pq, (-1,))
        ones = np.ones_like(pq)
        yp1 = np.concatenate((pq, ones), axis=1)
        if self.phase == Phase.TRAIN:
            # LSR solution of Q_i = k_1\hat{Q_i}+k_2. One can use the form of Eqn. (17) in the paper
            # However, for an efficient implementation, we use the matrix form of the solution here.
            # That is, h = (X^TX)^{-1}X^TY is the LSR solution of Y = Xh,
            # where X = [\hat{\mathbf{Q}}, \mathbf{1}], h = [k_1,k_2]^T, and Y=\mathbf{Q}.
            h = np.matmul(
                np.linalg.inv(np.matmul(yp1.transpose(), yp1)), np.matmul(yp1.transpose(), sq)
            )
            self.k[i] = h[0].item()
            self.b[i] = h[1].item()
        else:
            h = np.reshape(np.asarray([self.k[i], self.b[i]]), (-1, 1))
        pq = np.matmul(yp1, h)

        return np.reshape(pq, (-1,))


def dump_scalar_metrics(
    metrics: Dict, writer: SummaryWriter, phase: Phase, global_step: int = 0, dataset: str = ''
):
    prefix = phase.name.lower() + (f'_{dataset}' if dataset else '')
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f'{metric_name}/{prefix}',
            metric_value,
            global_step=global_step,
        )


def add_metrics_dict(metrics: dict, metrics_new: dict) -> dict:
    for key, value in metrics_new.items():
        metrics[key] = metrics.get(key, 0) + value
    return metrics


def divide_metrics(metrics: dict, n: int) -> dict:
    return {key: (value / n) for key, value in metrics.items()}


def compute_lpips(iqa_metric_computer: InferenceModel, x: torch.tensor, y: torch.tensor):
    return iqa_metric_computer(x, y)
