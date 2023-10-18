from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn


class Attacker(ABC):
    """A base class for untargeted adversarial attacks on images.

    :param model: The model to be attacked.
    :param scaler: The gradient scaler.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def run(self, inputs: Tensor, target: Tensor):
        """Perform the untargeted attack.

        :param inputs: Benign image batch to attack. Must have shape
            (batch size, 3, height, width) and values in range [0..1].
        :param target: Ground truth labels in one-hot format. The attacker's
            goal is to push the predictions *away* from ground truth.
        :returns: The attacked image batch. Some implementations may return
            additional information such as predictions and loss values.

        Both inputs and target must be on device (.cuda()).
        """


@dataclass
class AttackerOutput:
    inputs_adv: Tensor
    preds: Tensor
    loss: Tensor
