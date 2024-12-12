from neural_lyapunov_training.SixDOF_Quadrotor import SixDOFQuadrotorDynamics
from neural_lyapunov_training.dynamical_system import SecondOrderDiscreteTimeSystem, IntegrationMethod
import scipy
import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
# import pybullet as p
from math import ceil

# torch default options
set_t = {
    "dtype": torch.float32,
    "device": torch.device("cpu"),  # set to cpu if you don't have a graphics card
}
save_path = "6dof_quadrotor_plots/"

# integration methods for state evolution
position_integration, velocity_integration = IntegrationMethod.ExplicitEuler, IntegrationMethod.ExplicitEuler

# spelled out for clearer context
dynamic_parameters = {
    "mass": 0.15,
    "length": 0.5,
    "damping": 0.1,
    "time_step": 0.01,
    "time_horizon": 10
}

to_numpy = lambda x: x.detach().cpu().numpy()  # converts Tensor to Numpy array

# basic neural network architecture
class SimpleNNController(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):

        super(SimpleNNController, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.model(x)

def main(show, save):

    # get dynamic_parameters
