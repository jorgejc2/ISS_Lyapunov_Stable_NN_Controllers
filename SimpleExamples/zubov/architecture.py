import torch
import torch.nn as nn
from torch import Tensor, optim, tanh
from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict, List

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

class ZubovNetwork(nn.Module):

    def __init__(self, dynamical_system, lambda_b: float, lb: Tensor, ub: Tensor, alpha: float, width: int,
                 n_layers: int, input_dim: int, lrate: float,
                 horizon:float=1200, weight_decay:float=1e-5):
        """

        :param dynamical_system:
        :param lambda_b:
        :param lb:              lower limit of sampling region 𝒟
        :param ub:              upper limit of sampling region 𝒟
        :param alpha:
        :param width:           width of each layer
        :param n_layers:        number of NN layers
        :param input_dim:       state input dimension
        :param lrate:           gradient descent learning rate
        :param horizon:         time horizon to use when calculating trajectories
        :param weight_decay:    gradient descent weight decay
        """
        super().__init__()

        # create the network
        layers = []
        for i in range(n_layers - 1):
            in_dim = input_dim if i == 0 else width
            layers.extend([
                (f"Linear_{i}", nn.Linear(in_dim, width)),
                (f"Activation_{i}", nn.ReLU())
            ])
        # append the last layer
        layers.extend([
            (f"Linear_{i}", nn.Linear(width, 1)),
            (f"Activation_{i}", nn.Sigmoid())  # output of Zubov W(x) ∈ [0,1]
        ])
        self.model = nn.Sequential(OrderedDict(layers))
        self.lrate = lrate
        self.weight_decay = weight_decay
        self.dynamical_system = dynamical_system
        self.lambda_b = lambda_b
        self.lb = lb
        self.ub = ub
        self.alpha = alpha
        self.horizon = horizon
        self.num_steps = int(horizon / self.dynamical_system.dt)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=weight_decay)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @staticmethod
    def maximal_lyapunov_fn(traj: Tensor) -> Tensor:
        """
        The true maximal Lyapunov function takes value:

        * ∫₀ᴴ ‖x(t;x₀)‖ dt where H = ∞ if the initial state x₀ is stabilizable
        * ∞ if the initial state x₀ is not stabilizable

        Since it is difficult to calculate this integral, we instead perform a discrete calculation
        that serves as an estimate of the true maximal Lyapunov function definition.
        This function returns the value:

        ∑ᵢ₌₀ᴺ ‖x(i⋅Δt; x₀)‖

        where Δt is the discrete time step of our system, and H = Δt⋅N where 1 < N < ∞ or the
        number of discrete time steps to run the trajectory for.

        :param traj:
        :return:
        """
        traj_norm = torch.linalg.vector_norm(traj, dim=2)
        lya_discrete = traj_norm.sum(dim=1)

        return lya_discrete

    @staticmethod
    def project_to_box_border(x: Tensor, x_L: Tensor, x_U: Tensor) -> Tensor:
        """

        :param x:       box samples
        :param x_L:     lower limit box to project onto
        :param x_U:     upper limit box to project onto
        :return:
        """

        # reshape and broadcast limits along batch dimension
        x_L = x_L.reshape(1, -1).expand(x.shape[0], -1)
        x_U = x_U.reshape(1, -1).expand(x.shape[0], -1)
        b = torch.arange(x.shape[0])  # used for indexing

        x = torch.clamp(x, min=x_L, max=x_U)
        min_l, min_l_idx = torch.min((x - x_L).abs(), dim=1)
        min_u, min_u_idx = torch.min((x - x_U).abs(), dim=1)

        mask = min_l < min_u
        min_idx = torch.where(mask, min_l_idx, min_u_idx)
        border = torch.where(mask, x_L[b, min_l_idx], x_U[b, min_u_idx])

        x[b, min_idx] = border[b]  # perform projection

        return x


    def step(self, x: Tensor) -> Tuple[float, List[float]]:
        """

        :param x: (batch_size, input_dim)
        :return:
        """
        # function to calculate gradients of y w.r.t. x
        def _gradient(x: Tensor, y: Tensor, grad_outputs=None):
            if grad_outputs is None:
                grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
            return grad
        psi = lambda t : torch.linalg.vector_norm(t, dim=1)

        # Get points on the border of the region of interest, ∂R₂ where R₂ = {αx : x ∈ R₁},
        # R₁ ⊆ 𝒟, and 𝒟 is the region from which we are drawing samples.
        x_b = ZubovNetwork.project_to_box_border(self.alpha*x, self.lb, self.ub)

        # zero the gradients
        self.optimizer.zero_grad()

        yhat = self.forward(x)  # zubov output
        grad_output = _gradient(x, yhat) # grad of zubov w.r.t. x

        with torch.no_grad():
            # in case dynamical system is described as another NN, we do not want to update its parameters
            fx = self.dynamical_system(x)  # return the next state
            traj = self.dynamical_system(x, self.num_steps)  # return the trajectory

        max_v = ZubovNetwork.maximal_lyapunov_fn(traj)

        loss_z = (self.forward(torch.zeros((1, self.input_dim)))**2).squeeze(0)
        loss_r = torch.linalg.vector_norm((yhat - self.alpha * max_v)**2, dim=0, ord=1)
        loss_p1 = torch.einsum('bi,bi->b', grad_output, fx)
        loss_p2 = self.alpha * (1 - yhat) * (1 + yhat) * psi(x)
        loss_p = torch.linalg.vector_norm((loss_p1 + loss_p2)**2, dim=0, ord=1)
        loss_b = self.lambda_b * torch.linalg.vector_norm((self.forward(x_b)).abs(), dim=0, ord=1)

        loss = loss_z + loss_r + loss_p + loss_b

        # perform backward gradient calculations
        loss.backward()

        return loss.item(), [t.item() for t in [loss_z, loss_r, loss_p]]

