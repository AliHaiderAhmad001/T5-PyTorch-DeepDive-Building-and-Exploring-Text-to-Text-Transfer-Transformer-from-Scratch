import numpy as np
import torch.optim as optim
from torch.optim.optimizer import Optimizer

class ScheduledOptim:
    """
    A wrapper class around an optimizer to manage learning rate scheduling.
    
    This implements an "inverse square root" schedule with a warm-up phase,
    which sets a constant learning rate for the first N steps, followed by a
    decay phase where the learning rate decreases according to the inverse square root
    of the step number.

    Attributes:
        optimizer (Optimizer): The wrapped optimizer.
        d_model (int): The dimensionality of the model's hidden layers.
        n_warmup_steps (int): The number of steps to apply the warm-up.
        n_current_steps (int): The current step in the optimization process.

    Methods:
        step_and_update_lr(): Performs a single optimization step and updates
            the learning rate according to the schedule.
        zero_grad(): Clears the gradients of all optimized parameters.
        calculate_lr(): Computes the current learning rate according to the
            schedule.
    """
    def __init__(self, optimizer: 'Optimizer', d_model: int, n_warmup_steps: int = 10000) -> None:
        """
        Initializes the ScheduledOptim instance.

        Args:
            optimizer (Optimizer): The optimizer to wrap.
            d_model (int): The dimensionality of the model's hidden layers.
            n_warmup_steps (int): The number of steps over which to linearly increase the learning rate.
        """
        self.optimizer: 'Optimizer' = optimizer
        self.d_model: int = d_model
        self.n_warmup_steps: int = n_warmup_steps
        self.n_current_steps: int = 0

    def step_and_update_lr(self) -> None:
        """
        Performs an optimization step and updates the learning rate according
        to the current step using an inverse square root schedule.
        """
        self.n_current_steps += 1
        lr: float = self.calculate_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self) -> None:
        """
        Clears the gradients of all optimized parameters by calling zero_grad
        on the wrapped optimizer.
        """
        self.optimizer.zero_grad()

    def calculate_lr(self) -> float:
        """
        Calculates the learning rate using an "inverse square root" schedule
        with a warm-up phase.

        Returns:
            float: The calculated learning rate.
        """
        factor: float = self.d_model ** (-0.5)
        step: int = self.n_current_steps
        warmup: int = self.n_warmup_steps

        if step < warmup:
            lr: float = factor * (warmup ** 0.5)
        else:
            lr: float = factor * (step ** -0.5) * (warmup ** 0.5)
        
        return lr
