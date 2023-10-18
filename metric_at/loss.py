import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = 1e-8


class IQALoss(nn.Module):
    def __init__(
        self,
        loss_type: str = 'linearity',
        alpha: list[float] = [1, 0],
        beta: list[float] = [0.1, 0.1, 1],
        p: float = 2.0,
        q: float = 2.0,
        monotonicity_regularization: bool = True,
        gamma: float = 0.1,
        detach: bool = False,
    ) -> nn.Module:
        super(IQALoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.monotonicity_regularization = monotonicity_regularization
        self.gamma = gamma
        self.detach = detach

    def forward(self, y_pred, y):
        y = y[0].view(-1, 1)
        loss = 0
        if self.beta[-1] > 0:
            loss += self.beta[-1] * self.loss_func(y_pred[-1], y)
        if self.beta[0] > 0:
            loss += self.beta[0] * self.loss_func(y_pred[0], y)
        if self.beta[1] > 0:
            loss += self.beta[1] * self.loss_func(y_pred[1], y)

        return loss

    def loss_func(self, y_pred, y):
        if self.loss_type == 'mae':
            loss = F.l1_loss(y_pred, y)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(y_pred, y)
        elif self.loss_type == 'norm-in-norm':
            loss = norm_loss_with_normalization(
                y_pred, y, alpha=self.alpha, p=self.p, q=self.q, detach=self.detach
            )
        elif self.loss_type == 'min-max-norm':
            loss = norm_loss_with_min_max_normalization(
                y_pred, y, alpha=self.alpha, detach=self.detach
            )
        elif self.loss_type == 'mean-norm':
            loss = norm_loss_with_mean_normalization(
                y_pred, y, alpha=self.alpha, detach=self.detach
            )
        elif self.loss_type == 'scaling':
            loss = norm_loss_with_scaling(y_pred, y, alpha=self.alpha, p=self.p, detach=self.detach)
        else:
            loss = linearity_induced_loss(y_pred, y, self.alpha, detach=self.detach)
        if self.monotonicity_regularization:
            loss += self.gamma * monotonicity_regularization(y_pred, y, detach=self.detach)
        return loss


def monotonicity_regularization(y_pred, y, detach=False):
    """monotonicity regularization"""
    if y_pred.size(0) > 1:  #
        ranking_loss = F.relu((y_pred - y_pred.t()) * torch.sign((y.t() - y)))
        scale = 1 + torch.max(ranking_loss.detach()) if detach else 1 + torch.max(ranking_loss)
        return torch.sum(ranking_loss) / y_pred.size(0) / (y_pred.size(0) - 1) / scale
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_normalization(
    y_pred: torch.tensor,
    y: torch.tensor,
    alpha: list[float] = [1, 1],
    p: float = 2.0,
    q: float = 2.0,
    detach: bool = False,
    exponent: bool = True,
) -> torch.tensor:
    """norm_loss_with_normalization: norm-in-norm"""
    batch_size = y_pred.size(0)
    if batch_size > 1:
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)
        y_pred = y_pred / (eps + normalization)
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1, 1.0 / q)) * np.power(
            batch_size, max(0, 1.0 / p - 1.0 / q)
        )  # p, q>0
        loss0, loss1 = 0, 0

        if alpha[0] > 0:
            err = y_pred - y
            # avoid gradient explosion when 0 <= p <1;
            # and avoid vanishing gradient problem when p < 0
            if p < 1:
                err += eps
            loss0 = torch.norm(err, p=p) / scale  # Actually, p = q = 2 is related to PLCC
            loss0 = torch.pow(loss0, p) if exponent else loss0  #
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())
            err = rho * y_pred - y
            # avoid gradient explosion when 0 <= p <1;
            # and avoid vanishing gradient problem when p < 0
            if p < 1:
                err += eps
            loss1 = torch.norm(err, p=p) / scale  # Actually, p = q = 2 is related to LSR
            loss1 = torch.pow(loss1, p) if exponent else loss1  #  #
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def linearity_induced_loss(y_pred, y, alpha=[1, 1], detach=False):
    """linearity-induced loss, actually MSE loss with z-score normalization"""
    if y_pred.size(0) > 1:  # z-score normalization: (x-m(x))/sigma(x).
        sigma_hat, m_hat = (
            torch.std_mean(y_pred.detach(), unbiased=False)
            if detach
            else torch.std_mean(y_pred, unbiased=False)
        )
        y_pred = (y_pred - m_hat) / (sigma_hat + eps)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + eps)
        scale = 4
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / scale  # ~ 1 - rho, rho is PLCC
        if alpha[1] > 0:
            rho = torch.mean(y_pred * y)
            loss1 = (
                F.mse_loss(rho * y_pred, y) / scale
            )  # 1 - rho ** 2 = 1 - R^2, R^2 is Coefficient of determination
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_min_max_normalization(y_pred, y, alpha=[1, 1], detach=False):
    if y_pred.size(0) > 1:
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - m_hat) / (eps + M_hat - m_hat)  # min-max normalization
        y = (y - torch.min(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y)
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y)
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_mean_normalization(y_pred, y, alpha=[1, 1], detach=False):
    if y_pred.size(0) > 1:
        mean_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - mean_hat) / (eps + M_hat - m_hat)  # mean normalization
        y = (y - torch.mean(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_scaling(y_pred, y, alpha=[1, 1], p=2, detach=False):
    if y_pred.size(0) > 1:
        normalization = torch.norm(y_pred.detach(), p=p) if detach else torch.norm(y_pred, p=p)
        y_pred = y_pred / (eps + normalization)  # mean normalization
        y = y / (eps + torch.norm(y, p=p))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.
