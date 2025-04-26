"""
Additional utilities to aid in neural network training in PyTorch.
"""
import torch.nn as nn
import torch.optim as optim
from functools import partial
from typing import *

### Custom LR Schedulers
def linear_decay(epoch, initial_lr, final_lr, total_epochs, last_decay:float=1e-3):
    """

    :param epoch:
    :param initial_lr:
    :param final_lr:
    :param total_epochs:
    :param last_decay:
    :return:
    """
    # FIXME: Currently hard-coded to a tailored configuration that shows stable convergence. The parameters should
    # be modified instead of hard-coded.
    if epoch < 25:
        total_epochs = 25
        return 1 - epoch / total_epochs * (1 - final_lr / initial_lr)
    else:
        return last_decay

def get_lr_scheduler(scheduler_type: str, optimizer: optim.Optimizer, lr_parameters: Dict[str, Any]):
    """
    Return the learning rate scheduler to use from a common set.
    :param scheduler_type:
    :param optimizer:
    :param lr_parameters:
    :return:
    """
    # set linear LR scheduler
    scheduler = None
    if scheduler_type == 'step':
        step_size = lr_parameters.pop('step_size', None)
        gamma = lr_parameters.pop('gamma', None)
        assert step_size is not None and gamma is not None, "Must specify step size and gamma to use Step LR"
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                   gamma=gamma, **lr_parameters)
    elif scheduler_type == 'linear':
        decay_func = partial(linear_decay, **lr_parameters)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_func)
    elif scheduler_type == 'none':
        pass
    else:
        raise ValueError(f"Scheduler type of {scheduler_type} is not recognized")

    return scheduler