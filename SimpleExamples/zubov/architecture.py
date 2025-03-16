import torch
import torch.nn as nn
from torch import Tensor, optim
from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict, List, Callable

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

class ZubovNetwork(nn.Module):

    def __init__(self, dynamical_system, lambda_b: float, lb: Tensor, ub: Tensor, alpha: float, width: int,
                 n_layers: int, input_dim: int, lrate: float,
                 horizon:float=1200, weight_decay:float=1e-5, clip_gradient_norm: Optional[float] = None):
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
        self.input_dim = input_dim
        self.alpha = alpha
        self.horizon = horizon
        self.num_steps = int(horizon / self.dynamical_system.dt)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=weight_decay)
        self.clip_gradient_norm = clip_gradient_norm

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def forward_with_coords(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """

        Before the forward pass, clone the input and enable its gradient. Returning this cloned input allows the
        output of the network to be differentiated w.r.t. the input.

        :param x: (batches, input_dim)
        :return:
        """
        x = x.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        output = self.forward(x)

        return output, x

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
        lya_discrete = traj_norm.sum(dim=1, keepdim=True)

        return lya_discrete

    @staticmethod
    def project_to_box_border(x: Tensor, x_L: Tensor, x_U: Tensor) -> Tensor:
        """

        :param x:       box samples
        :param x_L:     lower limit box to project onto
        :param x_U:     upper limit box to project onto
        :return:
        """

        # For points inside the box, project to the nearest boundary
        x_L = x_L.reshape(1, -1).expand(x.shape[0], -1)
        x_U = x_U.reshape(1, -1).expand(x.shape[0], -1)
        b = torch.arange(x.shape[0])

        # x = torch.clamp(x, min=x_L, max=x_U)
        min_l, min_l_idx = torch.min((x - x_L).abs(), dim=1)
        min_u, min_u_idx = torch.min((x - x_U).abs(), dim=1)

        mask = min_l < min_u
        min_idx = torch.where(mask, min_l_idx, min_u_idx)
        border = torch.where(mask, x_L[b, min_l_idx], x_U[b, min_u_idx])

        x[b, min_idx] = border  # perform projection

        return x


    def step(self, x: Tensor, x_b: Tensor, u: Callable) -> Tuple[float, List[float]]:
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

        psi = lambda t : torch.linalg.vector_norm(t, dim=1, keepdim=True)

        # Get points on the border of the region of interest, ∂R₂ where R₂ = {αx : x ∈ R₁},
        # R₁ ⊆ 𝒟, and 𝒟 is the region from which we are drawing samples.
        # x_b = ZubovNetwork.project_to_box_border(self.alpha*x, self.lb, self.ub)

        with torch.no_grad():
            # in case dynamical system is described as another NN, we do not want to update its parameter
            # fx = self.dynamical_system(x)  # return the next state
            traj = self.dynamical_system(x, u, self.num_steps)  # return the trajectory
            fx = traj[:, 0, :]  # get the next state
            max_v = ZubovNetwork.maximal_lyapunov_fn(traj)
            psi_output = psi(x)

        # zero the gradients
        self.optimizer.zero_grad()

        yhat, coords = self.forward_with_coords(x)  # zubov output

        grad_output = _gradient(coords, yhat) # grad of zubov w.r.t. x

        # Loss Z ensures that at the equilibrium, the output of W(x) is 0.
        loss_z = (self.forward(torch.zeros((1, self.input_dim)))**2)
        # Loss R ensures that W(x) = tanh(a*V(x)) where V(x) in this implementation is
        # an approximation of the defined Maximal Lyapunov function
        loss_r = (yhat - torch.tanh(self.alpha * max_v))**2
        # Loss P is the derivative condition: ∂ₓW(x)ᵀf(x) = a(1−W(x))(1+W(x))Φ(x)
        # where Φ(x) := ‖x‖
        loss_p1 = torch.einsum('bi,bi->b', grad_output, fx).unsqueeze(1)
        loss_p2 = self.alpha * (1 - yhat) * (1 + yhat) * psi_output
        loss_p = (loss_p1 + loss_p2)**2
        loss_b = self.lambda_b * (self.forward(x_b) - 1).abs()
        loss = loss_z + loss_r + loss_p + loss_b
        total_loss = loss.mean()

        # perform backward gradient calculations
        total_loss.backward()

        # perform gradient clipping
        # typically recommended for stable training
        if self.clip_gradient_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_gradient_norm)

        # update the model parameters
        self.optimizer.step()

        rest_losses = [t.mean().item() if isinstance(t, Tensor) else t for t in [loss_z, loss_r, loss_p, loss_b]]
        return total_loss.item(), *rest_losses


if __name__ == "__main__":
    # FIXME: Clean up, solely for debugging

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    to_numpy = lambda x: x.detach().cpu().numpy()


    def plot_2d_box(ax, x_L, x_U):
        width = x_U[0] - x_L[0]
        height = x_U[1] - x_L[1]

        rect = patches.Rectangle(x_L, width, height, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.set_xlim(x_L[0] - 1, x_U[0] + 1)
        ax.set_ylim(x_L[1] - 1, x_U[1] + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
    pendulum_continuous = lambda x : x
    x_L = -torch.ones((2,))
    x_U = torch.ones((2,))
    # zubov_args = {
    #     "dynamical_system": pendulum_continuous,
    #     "lambda_b": 0.01,
    #     "lb": x_L,
    #     "ub": x_U,
    #     "alpha": 0.2,
    #     "width": 32,
    #     "n_layers": 5,
    #     "input_dim": 2,
    #     "lrate": 0.01,
    #     "horizon": 20,
    #     "weight_decay": 1e-5
    # }
    # zubov_network = ZubovNetwork(**zubov_args)
    x = torch.rand(2048, 2)
    x_new = ZubovNetwork.project_to_box_border(x, x_L, x_U)

    x_np = to_numpy(x)
    x_new_np = to_numpy(x_new)
    print(f"x: {x_np}")
    print(f"x_new: {x_new_np}")

    fig, ax = plt.subplots()
    print(ax)
    plot_2d_box(ax, x_L, x_U)
    ax.scatter(x_np[:, 0], x_np[:, 1], color='red', label='Original Points')
    ax.scatter(x_new_np[:, 0], x_new_np[:, 1], color='green', label='Projected Points')
    ax.legend()
    plt.show()