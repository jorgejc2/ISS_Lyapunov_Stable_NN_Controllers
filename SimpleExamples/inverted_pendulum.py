from neural_lyapunov_training.pendulum import PendulumDynamics
from neural_lyapunov_training.dynamical_system import SecondOrderDiscreteTimeSystem, IntegrationMethod
import scipy
import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
from math import ceil

# torch default options
set_t = {
    "dtype": torch.float32,
    "device": torch.device("cuda"),  # set to cpu if you don't have a graphics card
}
save_path = "inverted_pendulum_plots/"

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

def plot_states(
        t: Union[Tensor, ndarray],
        x: [Tensor, ndarray],
        title: str,
        y_labels: Optional[list[str]] = None,
        equilibrium_points: Optional[Union[Tensor, ndarray]] = None,
        show: bool = False,
        save_name: Optional[str] = None,
        max_cols: int = 4,
        max_subplots: int = 12
):
    """
    Plots the state trajectories for some dynamical system.
    :param t:               Array of timesteps
    :param x:               State trajectories, 1st dim is time, 2nd dim are states
    :param title:           The suptitle of the plot
    :param y_labels:        Descriptors for the states
    :param show:            True if the plot should be shown
    :param save_name:       If not None, saves the plot with the specified file name
    :param max_cols:        Maximum number of columns in the plot
    :param max_subplots:    Maximum number of subplots to handle
    :return:
    """

    assert len(x.shape) == 2, "States should be two dimensional arrays"

    # Convert to numpy arrays
    if isinstance(t, Tensor):
        t = to_numpy(t)
    if isinstance(x, Tensor):
        x = to_numpy(x)
    if equilibrium_points is not None and isinstance(equilibrium_points, Tensor):
        equilibrium_points = to_numpy(equilibrium_points)

    # handles subplot formatting for arbitrary number of state dimensions
    n_dim = x.shape[1]
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)
    cols = max_cols
    plot_batches = min(n_dim, max_subplots)
    if plot_batches <= cols:
        cols = plot_batches
        rows = 1
    else:
        rows = ceil(plot_batches / cols)
    # subplot offset where first digit are total rows, second digit are total columns, third digit is current subplot
    base_subplot = rows * 100 + cols * 10
    axes = [(i, fig.add_subplot(base_subplot + i + 1)) for i in range(n_dim)]

    for i, ax in axes:
        ax.plot(t, x[:, i])
        ax.set_xlabel("Time")
        if y_labels is not None:
            ax.set_ylabel(y_labels[i])
        else:
            ax.set_ylabel(f"State {i}")
        if equilibrium_points is not None:
            ax.axhline(y=equilibrium_points[i], color='r', linestyle='--', linewidth=1, label='Equilibrium')
            ax.legend()
        ax.grid()

    plt.tight_layout()  # resize the subplots

    if save_name is not None:
        fig.savefig(save_path + save_name)

    if show:
        plt.show()
    else:
        plt.close()

    return

def linearize_pendulum(pendulum_continuous: PendulumDynamics):
    """
    Uses PyTorch autograd to linearize the pendulum about the equilibrium point.
    This is one alternative to exactly computing the linearization.
    :param pendulum_continuous: Continuous pendulum model
    :return:
    """
    x = torch.tensor([[0.0, 0.0]])
    x.requires_grad = True
    u = torch.tensor([[0.0]])
    u.requires_grad = True
    qddot = pendulum_continuous.forward(x, u)
    A = torch.empty((2, 2))
    B = torch.empty((2, 1))
    A[0, 0] = 0
    A[0, 1] = 1
    B[0, 0] = 0
    A[1], B[1] = torch.autograd.grad(qddot[0, 0], [x, u])
    return A, B

def compute_lqr(pendulum_continuous: PendulumDynamics):
    """
    Computes the LQR controller about some equilibrium point.
    The solution entails a controller in linear feedback form u=Kx
    as well a Lyapunov function in quadratic form V=x^TSx.
    :param pendulum_continuous: Continuous pendulum model
    :return:                    Controller, Lyapunov function
    """
    A, B = linearize_pendulum(pendulum_continuous)
    A_np, B_np = A.detach().numpy(), B.detach().numpy()
    Q = np.eye(2)
    R = np.eye(1) * 100
    S = scipy.linalg.solve_continuous_are(A_np, B_np, Q, R)
    K = -np.linalg.solve(R, B_np.T @ S)
    return K, S

def approximate(
        system: nn.Module,
        system_input: Tensor,
        target: Tensor,
        lr: float,
        max_iter: int
):
    """
    Fits a neural network controller to an LQR controller
    :param system:          The NN controller to train
    :param system_input:    State samples
    :param target:          Target control for each state sample
    :param lr:              Learning rate
    :param max_iter:        Number of training iterations
    :return:
    """
    optimizer = torch.optim.Adam(system.parameters(), lr=lr)
    losses = np.zeros(max_iter)
    for i in range(max_iter):
        optimizer.zero_grad()
        output = torch.nn.MSELoss()(system.forward(system_input), target)
        print(f"iter {i}, loss {output.item()}")
        losses[i] = output.item()
        output.backward()
        optimizer.step()

    return losses

def main(show):

    # get dynamic parameters
    m = dynamic_parameters["mass"]
    l = dynamic_parameters["length"]
    beta = dynamic_parameters["damping"]
    dt = dynamic_parameters["time_step"]
    T = dynamic_parameters["time_horizon"]

    # initialize the dynamical system
    pendulum_continuous = PendulumDynamics(m=m, l=l, beta=beta)
    dynamics = SecondOrderDiscreteTimeSystem(
        pendulum_continuous,
        dt=dt,
        position_integration=position_integration,
        velocity_integration=velocity_integration,
    )

    n_dim = pendulum_continuous.nx  # state dimension
    m_dim = pendulum_continuous.nu  # controller dimension

    time_steps = torch.arange(0, T, dt)  # time steps
    n_time_steps = len(time_steps)

    ## Shows how the system evolves with constant epsilon disturbance in the controller
    states = torch.zeros((n_time_steps, n_dim), **set_t)
    u = torch.full((n_time_steps, m_dim), torch.finfo(set_t["dtype"]).eps, **set_t)
    # simulate dynamics
    for i in range(1, n_time_steps):
        time = time_steps[i]
        prev_state = states[i-1].unsqueeze(0)
        control = u[i-1].unsqueeze(0)
        new_state = dynamics.forward(prev_state, control).squeeze()
        states[i] = new_state
        print(f"time {time:.2f} sec | state {to_numpy(new_state)} | input {to_numpy(control)}")

    # plot results
    plot_states(time_steps, states, r"State Trajectories with $\epsilon$ Control Disturbance",
                [r"$\theta$", r"$\dot{\theta}$"], pendulum_continuous.x_equilibrium,
                show, "const_perturbed_control.png")

    ## Shows how the system may be stabilized via LQR
    K, S = compute_lqr(pendulum_continuous)
    # convert to Tensors
    K = torch.from_numpy(K).to(**set_t)  # linear feedback controller
    S = torch.from_numpy(S).to(**set_t)  # quadratic Lyapunov function
    x = (torch.rand((100, 2), **set_t) - 0.5) * 2  # create a bunch of state samples near the origin to stabilize
    V = torch.sum(x * (x @ S), axis=1, keepdim=True)
    print(f"K shape: {K.shape}, V shape: {V.shape}")

    # plot the output of the Lyapunov function for these samples
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_np = np.linspace(-1, 1, 100)
    y_np = np.linspace(-1, 1, 100)
    x_np, y_np = np.meshgrid(x_np, y_np)
    lqr_lya = lambda x: np.sum(x * (x @ to_numpy(S)), axis=1)  # applies Lyapunov function to state inputs
    V_np = lqr_lya(np.concatenate([x_np.ravel().reshape(-1, 1), y_np.ravel().reshape(-1, 1)], axis=1))
    V_np = V_np.reshape(x_np.shape)
    ax.plot_wireframe(x_np, y_np, V_np, color='b')
    ax.set_title("Lyapunov Function from LQR Solution")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_zlabel(r"$V(x)$")
    plt.savefig(save_path + "lqr_quadratic_lyapunov_function.png")
    plt.show() if show else plt.close()

    # now initialize the states with some disturbance
    disturbance = torch.tensor([1e-3, -1e-3], **set_t).reshape(1, -1)
    states = torch.ones((n_time_steps, n_dim), **set_t) * disturbance
    # simulate dynamics
    for i in range(1, n_time_steps):
        time = time_steps[i]
        prev_state = states[i - 1].unsqueeze(0)
        control = K @ prev_state.permute(1, 0)
        new_state = dynamics.forward(prev_state, control).squeeze()
        states[i] = new_state
        print(f"time {time:.2f} sec | state {to_numpy(new_state)} | input {to_numpy(control)}")

    # plot results
    plot_states(time_steps, states, r"State Trajectories with LQR",
                [r"$\theta$", r"$\dot{\theta}$"], pendulum_continuous.x_equilibrium,
                show, "lqr_control.png")

    ## Shows how the system may be stabilized using a NN that learns the LQR controller
    x = (torch.rand((100000, 2), **set_t) - 0.5) * 2  # create a bunch of state samples near the origin to stabilize
    lqr_samples = x @ K.permute(1, 0)  # get the control feedback for these states
    controller = SimpleNNController(n_dim, m_dim).to(device=set_t["device"])  # initialize a simple NN controller
    losses = approximate(controller, x, lqr_samples, lr=0.01, max_iter=1000)  # fit the NN to the LQR controller

    # plot the training loss
    plt.figure(figsize=(12, 8))
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("NN Controller Training Convergence")
    plt.grid()
    plt.savefig(save_path + "nn_controller_loss.png")
    plt.show() if show else plt.close()

    # now initialize the states with some disturbance
    states = torch.ones((n_time_steps, n_dim), **set_t) * disturbance
    # simulate dynamics
    for i in range(1, n_time_steps):
        time = time_steps[i]
        prev_state = states[i - 1].unsqueeze(0)
        control = controller(prev_state)
        new_state = dynamics.forward(prev_state, control).squeeze()
        states[i] = new_state
        print(f"time {time:.2f} sec | state {to_numpy(new_state)} | input {to_numpy(control)}")

    # plot results
    plot_states(time_steps, states, r"State Trajectories with NN Controller",
                [r"$\theta$", r"$\dot{\theta}$"], pendulum_continuous.x_equilibrium,
                show, "nn_control.png")

if __name__ == '__main__':
    show_plots = True
    main(show_plots)