# Adapted from Opacus optimizers

from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .optimizer import DPOptimizer, _generate_noise, _check_processed_flag, _mark_as_processed



class ObjPertOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    objective perturbation where the noise is the same at every iteration
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "sum",
        generator=None,
        secure_mode: bool = False,
    ):
        super(ObjPertOptimizer, self).__init__(
            optimizer=optimizer, 
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode
        )
        self.lambd = 0
        self.num_batches = 0
        for p in self.params:
            p.noise = _generate_noise(std=max_grad_norm*noise_multiplier, reference=p)

    def set_lambd(self, lambd):
        self.lambd = lambd

    def set_num_batches(self, num_batches):
        self.num_batches = num_batches


    def add_noise(self):
        """
        Adds noise (and regularization) to clipped gradients. Stores clipped, regularized and noised result in ``p.grad``
        """
        if self.lambd == 0:
            raise ValueError("Objective perturbation requires a positive lambda value. Use set_lambd() before training the model.")
        if self.num_batches == 0:
            raise ValueError("Number of batches is not set. Set num_batches = n_train//batch_size")
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = (p.summed_grad + self.lambd/self.num_batches * p + p.noise/self.num_batches).view_as(p.grad).clone()
            _mark_as_processed(p.summed_grad)
