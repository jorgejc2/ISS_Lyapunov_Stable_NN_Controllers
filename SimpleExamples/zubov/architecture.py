import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor, optim
from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict, List, Callable
from SimpleExamples.zubov.neural_building_blocks import *

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "cos": Cosine,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "softmax": lambda: nn.Softmax(dim=1),  # dim might need to be customized
    "identity": nn.Identity
}

def get_activation(name: str):
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation: {name}")

### Custom LR Schedulers
def linear_decay(epoch, initial_lr, final_lr, total_epochs, last_decay):
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

class GeneralNeuralNetwork(nn.Module):
    def __init__(self, width: int, n_layers: int, input_dim: int, output_dim: int,
                 act: str = 'relu', final_act: str = 'tanh',):
        """
        :param width:           width of each layer
        :param n_layers:        number of NN layers
        :param input_dim:       input dimension
        :param output_dim:       output dimension

        """
        inter_act = get_activation(act)
        super().__init__()

        # create the network
        layers = []
        for i in range(n_layers - 1):
            in_dim = input_dim if i == 0 else width
            layers.extend([
                (f"Linear_{in_dim}_{width}_{i}", nn.Linear(in_dim, width)),
                (f"Activation_{act}_{i}", inter_act)
            ])
        # append the last layer
        layers.extend([
            (f"Linear_{width}_{output_dim}_{i}", nn.Linear(width, output_dim)),
            # (f"Activation_{final_act}_{i}", nn.Tanh())
        ])
        # final activation is not part of the model as it is allowed to be
        # swapped after training
        self.final_act = final_act

        self.model = nn.Sequential(OrderedDict(layers))
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.final_act(self.model(x))
        # return self.model(x)

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

    @property
    def final_act(self):
        return self._final_act

    @final_act.setter
    def final_act(self, final_act: str):
        assert final_act in ['tanh', 'sigmoid']
        self._final_act = nn.Tanh() if final_act == 'tanh' else nn.Sigmoid()

class ZubovNetwork(nn.Module):

    def __init__(self, dynamical_system, zubov_fn: GeneralNeuralNetwork, controller: Union[GeneralNeuralNetwork, Callable],
                 lb: Tensor, ub: Tensor, alpha: float, lrate: float, num_epochs: int, barrier_scale: float=2.,
                 max_steps:int=1e7, weight_decay:float=1e-5, clip_gradient_norm: Optional[float] = None,
                 c1: float = 5e1, c2: float = 3e1, c3: float = 1e1, c4: float = 1e1, c5: float = 1e1,
                 train_controller: bool = False, step_size: Optional[int] = None, gamma: Optional[float] = None,
                 final_lrate: Optional[float] = None, scheduler_type: str = 'none'):
        """

        :param dynamical_system:    A function describing the discrete time dynamics of the system of interest.
        :param zubov_fn:            A NN to learn the Zubov function, W(x).
        :param controller:          A callable function (potentially a NN) to use as the controller of the system.
        :param lb:                  Lower limit of the sampling region 𝒟.
        :param ub:                  Upper limit of the sampling region 𝒟.
        :param alpha:               Exponential decay rate.
        :param lrate:               Gradient descent learning rate.
        :param num_epochs:          The number of epochs the training will run. This does not need to be correct, only
                                    used for the learning scheduler if a scheduler is used and requires it.
        :param barrier_scale:       How much to scale the box for samples when sampling for the barrier loss function.
        :param max_steps:           The maximum number of steps to simulate a trajectory for in the dynamical system.
        :param weight_decay:        Gradient descent weight decay.
        :param clip_gradient_norm:  If specified, the gradient norm of the gradient is clipped.
        :param c1:                  Weighted parameters on the loss enforcing 0 condition.
        :param c2:                  Weighted parameters on the loss enforcing W matches tanh(aV(x)).
        :param c3:                  Weighted parameters on the loss enforcing gradient of W to be similar to gradient of V(x).
        :param c4:                  Weighted parameters on the loss enforcing the barrier condition for expanding the Lyapunov function.
        :param c5:                  Weighted parameters on the controller loss.
        :param train_controller:    When True and the controller is a NN, the controller will be trained as well.
        :param step_size:           Step size for the step learning rate scheduler.
        :param gamma:               The exponential decay for the step learning rate scheduler.
        :param final_lrate:         The final learning rate for the linear learning rate scheduler.
        :param scheduler_type:      The type of scheduler to use, in types ['step', 'linear', 'None']
        """
        super().__init__()

        self.lrate = lrate
        self.weight_decay = weight_decay
        self.dynamical_system = dynamical_system
        self.zubov_fn = zubov_fn
        self.controller = controller
        # if specified and the controller is a NN, it will be trained
        self.train_controller = train_controller if isinstance(controller, nn.Module) else False
        self.lb, self.ub = lb, ub
        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.barrier_scale = barrier_scale # how much to scale the box for samples when sampling for the barrier loss function
        self.alpha = alpha
        self.max_steps = int(max_steps)
        self._use_barrier = True
        self.horizon = self.max_steps * self.dynamical_system.dt
        if self.train_controller:
            # update parameters for the Zubov function and NN controller
            self.model_parameters = list(self.zubov_fn.parameters()) + list(self.controller.parameters())
        else:
            # only update parameters for the Zubov function
            self.model_parameters = list(self.zubov_fn.parameters())
        self.optimizer = optim.Adam(self.model_parameters, lr=lrate, weight_decay=weight_decay)
        self.clip_gradient_norm = clip_gradient_norm

        # set linear LR scheduler
        self.scheduler = None
        if scheduler_type == 'step':
            assert step_size is not None and gamma is not None, "Must specify step size and gamma to use Step LR"
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                                             gamma=gamma)
        elif scheduler_type == 'linear':
            decay_func_siren = partial(linear_decay, initial_lr=lrate, final_lr=final_lrate,
                                       total_epochs=num_epochs, last_decay=1e-3)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=decay_func_siren)
        elif scheduler_type == 'none':
            pass
        else:
            raise ValueError(f"Scheduler type of {scheduler_type} is not recognized")

    @staticmethod
    def project_to_uniform_box_border(n: int, x_U: Tensor, scale=1., noise_var: float = 0.) -> Tensor:
        """
        Projects x to a uniform box border that may be scaled.
        :param n:           Number of samples to generate
        :param x_U:         Upper limits of the box (it is assumed that x_L = -x_U)
        :param scale:       Scale of the box
        :param noise_var:   The Gaussian variance of the noise to be added to the samples (by default, no noise is added)
        :return:
        """
        x_U = x_U.reshape(1, -1)
        input_dim = x_L.shape[1]
        x = torch.rand((n, input_dim)) * 2 - 0.5
        xnorm = (x.abs() / (x_U * scale)).max(dim=1, keepdim=True)
        x /= xnorm
        if noise_var > 0.:
            noise = torch.randn_like(x) * noise_var
            return x + noise
        else:
            return x

    @staticmethod
    def project_to_box_border(x: Tensor, x_L: Tensor, x_U: Tensor, scale=1., noise_var: float = 0.) -> Tensor:
        """
        Projects a batch of samples to the border of a (potentially nonuniform) box that may be scaled.
        :param x:       box samples (assumed to already be in the range [x_L, x_U])
        :param x_L:     lower limit box to project onto
        :param x_U:     upper limit box to project onto
        :param scale:       Scale of the box
        :param noise_var:   The Gaussian variance of the noise to be added to the samples (by default, no noise is added)
        :return:
        """

        # For points inside the box, project to the nearest boundary
        x_L = scale * x_L.reshape(1, -1).expand(x.shape[0], -1).to(x)
        x_U = scale * x_U.reshape(1, -1).expand(x.shape[0], -1).to(x)
        b = torch.arange(x.shape[0])

        # x = torch.clamp(x, min=x_L, max=x_U)
        min_l, min_l_idx = torch.min((x - x_L).abs(), dim=1)
        min_u, min_u_idx = torch.min((x - x_U).abs(), dim=1)

        mask = min_l < min_u
        min_idx = torch.where(mask, min_l_idx, min_u_idx)
        border = torch.where(mask, x_L[b, min_l_idx], x_U[b, min_u_idx])

        x[b, min_idx] = border  # perform projection

        if noise_var > 0.:
            noise = torch.randn_like(x) * noise_var
            return x + noise
        else:
            return x

    @staticmethod
    def jacobian(x: Tensor, y: Tensor, grad_outputs=None):
        """
        Calculates the Jacobian of the output with respect to input x.
        :param x:               Input tensor
        :param y:               Output tensor
        :param grad_outputs:    Tensor to save output that may be optionally pre-defined
        :return:                Jacobian of y w.r.t. x
        """
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad

    @staticmethod
    @torch.no_grad()
    def calculate_trajectory(x: Tensor, dynamical_system, controller: Union[GeneralNeuralNetwork, Callable],
                             norm_threshold: float, integ_threshold: float, max_steps: int):
        """
        Generates a trajectory for a batch of starting states and generates their output with respect to
        the maximal Lyapunov function which is defined as follows:

        The true maximal Lyapunov function takes value:

        * ∫₀ᴴ ‖x(t;x₀)‖ dt where H = ∞ if the initial state x₀ is stabilizable
        * ∞ if the initial state x₀ is not stabilizable

        Since it is difficult to calculate this integral, we instead perform a discrete calculation
        that serves as an estimate of the true maximal Lyapunov function definition.
        This function returns the value:

        ∑ᵢ₌₀ᴺ ‖x(i⋅Δt; x₀)‖*Δt

        where Δt is the discrete time step of our system, and H = Δt⋅N where 1 < N < ∞ or the
        number of discrete time steps to run the trajectory for.

        Stops when one of the following cases happen:

        * case one: norm falls below the norm threshold
        * case two: the trajectories stabilize
        * case three: the number of steps exceed max steps
        * case four: the Lyapunov output exceeds the integral threshold

        :param x:                   Initial batch of states
        :param dynamical_system:    Dynamical control system capable of generating trajectories
        :param controller:          Feedback controller
        :param norm_threshold:      Norm threshold used in case one
        :param integ_threshold:     Integral threshold used in case four
        :param max_steps:           Maximum length of the trajectories

        :return x_hist:     History of trajectories with shape  (batches, trajectory length, ndim)
        :return integ_acc:  Lyapunov output with shape (batches, 1)
        :return True/False: Boolean indicating whether the trajectories terminated early or not
        """
        # FIXME:
        #  This function does not work that well but is similar to how they implemented in their repo. Either this
        #  function is not correct, some parameters are bad, or this function is overall too complicated and not
        #  efficient compared to the 'maximal_lyapunov_fn' implementation.

        integ_acc = torch.zeros((x.shape[0], 1)).to(x)
        steps = 0
        x_hist = x.clone().unsqueeze(1) # (batches, trajectory length, ndim)
        while True:
            steps += 1
            norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
            integ_acc += norm * dynamical_system.dt

            case_one = (norm < norm_threshold).all().item()
            case_two = x_hist.shape[1] > 10 and (torch.linalg.vector_norm(x_hist[:, -1] - x_hist[:, -10], dim=1) < 1e-3).all().item()
            case_three = steps > max_steps
            case_four = (integ_acc > integ_threshold).all().item()

            if case_one or case_two or case_three:
                return x_hist, integ_acc, True
            elif case_four:
                return x_hist, integ_acc, False

            x = dynamical_system.forward(x, controller)
            x_hist = torch.cat([x_hist, x.unsqueeze(1)], dim=1)

    @staticmethod
    @torch.no_grad()
    def maximal_lyapunov_fn(alpha: float, zubov_fn: GeneralNeuralNetwork, traj: Tensor, dt: float,
                            improved_data_loss: bool = False) -> Tensor:
        """
        The true maximal Lyapunov function takes value:

        * ∫₀ᴴ ‖x(t;x₀)‖ dt where H = ∞ if the initial state x₀ is stabilizable
        * ∞ if the initial state x₀ is not stabilizable

        Since it is difficult to calculate this integral, we instead perform a discrete calculation
        that serves as an estimate of the true maximal Lyapunov function definition.
        This function returns the value:

        ∑ᵢ₌₀ᴺ ‖x(i⋅Δt; x₀)‖Δt

        where Δt is the discrete time step of our system, and H = Δt⋅N where 1 < N < ∞ or the
        number of discrete time steps to run the trajectory for.

        :param zubov_fn:            The Zubov function, W(x)
        :param traj:                A batch of trajectories
        :param dt:                  Discrete time step
        :param improved_data_loss:  Notice that the maximal Lyapunov function looks like a Value function in RL,
                                    therefore, we can separate out the Lyapunov function from time step 0 to T,
                                    and estimate the remaining terms.
        :return:
        """
        traj_norm = torch.linalg.vector_norm(traj, dim=2)
        traj_norm.sum(dim=1, keepdim=True) * dt
        if improved_data_loss:
            traj_norm += torch.arctanh(zubov_fn(traj[:, -1, :])) / alpha
            # next_wx = zubov_fn(traj[:, -1, :])
            # traj_norm += (1/(2*alpha))*torch.log((1 + next_wx)/(1 - next_wx))

        return traj_norm

    def step(self, x: Tensor, traj: Optional[Tensor] = None) -> Tuple[float, float, float, float, float, float]:
        """

        :param x: (batch_size, input_dim)
        :return total_loss:
        :return loss_z:
        :return loss_r:
        :return loss_p:
        :return loss_b:

        """

        # equivalent to −∇ₓV(x)ᵀf(x)
        psi = lambda t : torch.linalg.vector_norm(t, dim=1, keepdim=True)


        with torch.no_grad():
            # Get points on the border of the region of interest, ∂R₂ where R₂ = {αx : x ∈ R₁},
            # R₁ ⊆ 𝒟, and 𝒟 is the region from which we are drawing samples.
            # x_b = ZubovNetwork.project_to_box_border(x, self.lb, self.ub, scale=self.barrier_scale, noise_var=0.2)
            x_b = ZubovNetwork.project_to_box_border(x, self.lb, self.ub, scale=self.barrier_scale, noise_var=0.)
            # in case dynamical system is described as another NN, we do not want to update its parameter
            if traj is None:
                traj = self.dynamical_system.forward_trajectory(x, self.controller, self.max_steps)  # return the trajectory
            fx = traj[:, 0, :]  # get the next state
            lya_max = ZubovNetwork.maximal_lyapunov_fn(self.alpha, self.zubov_fn, traj, self.dynamical_system.dt,
                                                       improved_data_loss=True)
            psi_output = psi(x)

        yhat, coords = self.zubov_fn.forward_with_coords(x)  # zubov output
        grad_output = ZubovNetwork.jacobian(coords, yhat) # grad of zubov w.r.t. x

        ## CRITIC LOSS ##

        # Loss Z ensures that at the equilibrium, the output of W(x) is 0.
        loss_z = self.zubov_fn.forward(torch.zeros((1, self.dynamical_system.nx)).to(x))
        # loss_z = self.c1 * (loss_z**2).mean()
        loss_z = self.c1 * loss_z**2

        # Loss R ensures that W(x) = tanh(a*V(x)) where V(x) in this implementation is
        # an approximation of the defined Maximal Lyapunov function
        loss_r = yhat - torch.tanh(self.alpha * lya_max)
        # loss_r = self.c2 * (loss_r**2).mean()
        loss_r = self.c2 * loss_r**2

        # Loss P is the derivative condition: ∂ₓW(x)ᵀf(x) = a(1−W(x))(1+W(x))Φ(x)
        # where Φ(x) := ‖x‖
        loss_p1 = torch.einsum('bi,bi->b', grad_output, fx).unsqueeze(1)
        loss_p2 = self.alpha * (1 - yhat) * (1 + yhat) * psi_output
        loss_p = loss_p1 + loss_p2
        # loss_p = self.c3 * (loss_p**2).mean()
        loss_p = self.c3 * loss_p**2

        ## BARRIER LOSS ##
        if self._use_barrier:
            loss_b = self.zubov_fn.forward(x_b) - 1
            # loss_b = loss_b.abs().mean()
            loss_b = self.c4 * loss_b.abs()
        else:
            loss_b = 0.

        ## CONTROLLER LOSS ##
        if self.train_controller:
            grad_output /= torch.linalg.vector_norm(grad_output, dim=1, keepdim=True)
            loss_c = torch.einsum('bi,bi->b', grad_output.detach(), fx).unsqueeze(1)
            loss_c = self.c5 * loss_c**2
        else:
            loss_c = 0.

        ## FINAL LOSS ##
        # add up the total loss
        total_loss = (loss_z + loss_r + loss_p + loss_b + loss_c).mean()

        # zero the gradients
        self.optimizer.zero_grad()

        # perform backward gradient calculations
        total_loss.backward()

        # perform gradient clipping
        # typically recommended for stable training
        if self.clip_gradient_norm is not None:
            nn.utils.clip_grad_norm_(self.model_parameters, max_norm=self.clip_gradient_norm)

        # update the model parameters
        self.optimizer.step()

        rest_losses = [t.mean().item() if isinstance(t, Tensor) else t for t in [loss_z, loss_r, loss_p, loss_b, loss_c]]
        return total_loss.item(), *rest_losses

    def scheduler_step(self) -> Tuple[float, None]:
        """
        Steps the scheduler if it is being used.
        In addition, returns the siren learning rate before the scheduling step.
        :return:
        """

        # step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            last_lr = self.scheduler.get_last_lr()[0]
        else:
            last_lr = self.lrate

        return last_lr, None

    @property
    def use_barrier(self)->bool:
        return self._use_barrier

    @use_barrier.setter
    def use_barrier(self, use_barrier: bool):
        self._use_barrier = use_barrier

# class ZubovNetworkWithSiren(nn.Module):
#     def __init__(self, dynamical_system, lambda_b: float, lb: Tensor, ub: Tensor, alpha: float,
#                  input_dim: int, hidden_features: int, hidden_layers: int, output_dim: int,
#                  siren_lrate: float, latent_lrate: float, num_epochs: int, horizon: float = 1200,
#                  final_siren_lrate: Optional[float] = None, final_latent_lrate: Optional[float] = None,
#                  first_omega_0: int = 30, hidden_omega_0: float = 30., latent_dim: int = 0,
#                  step_size: Optional[int] = None, gamma: Optional[float] = None,
#                  c1: float = 5e1, c2: float = 3e3, c3: float = 1e2,
#                  clip_gradient_norm: Optional[float] = None, scheduler_type: str = 'none'
#         ):
#         """
#
#         Initializes a Siren model for fitting weak signed distance functions. Latent variables are supported as
#         well for modulation.
#
#         :param in_features:         Input dimension
#         :param hidden_features:     Hidden layer width
#         :param hidden_layers:       Number of hidden layers
#         :param out_features:        Output dimension
#         :param siren_lrate:         Learning rate for Siren network
#         :param latent_lrate:        Learning rate for latent variable parameters
#         :param first_omega_0:       omega to use for first Siren layer
#         :param hidden_omega_0:      omegas to use for intermediate Siren layers
#         :param latent_dim:          Dimension of the latent variable
#         :param step_size:           Number of steps before applying LR Scheduler
#         :param gamma:               LR Scheduler Decay
#         :param c1:                  First penalization parameter for Eikonal loss function (reference Siren paper for more details)
#         :param c2:                  Second penalization parameter for Eikonal loss function (reference Siren paper for more details)
#         :param c3:                  Third penalization parameter for Eikonal loss function (reference Siren paper for more details)
#         :param clip_gradient_norm:  Max norm to clip model gradients. Helps with stabilization when using latent variables.
#         """
#         super().__init__()
#         if final_siren_lrate is None: final_siren_lrate = siren_lrate
#         if final_latent_lrate is None: final_latent_lrate = latent_lrate
#         self.hidden_layers = hidden_layers
#         self.clip_gradient_norm = clip_gradient_norm
#         self.latent, self.modulator = None, None
#         self.c1, self.c2, self.c3 = c1, c2, c3
#         self.has_latent = latent_dim > 0
#         self.dynamical_system = dynamical_system
#         self.lambda_b = lambda_b
#         self.lb = lb
#         self.ub = ub
#         self.alpha = alpha
#         self.horizon = horizon
#         self.num_steps = int(horizon / self.dynamical_system.dt)
#         self.input_dim = input_dim
#         # optimizable parameters
#         self.opt_latent_parameters = []
#         self.opt_siren_parameters = []
#
#         # If using modulation, instantiate a modulator network and an optimizable latent tensor
#         # The modulator network and latent tensor are optimized separately from the Siren network for more
#         # fine-grained control
#         if latent_dim > 0:
#             self.modulator = Modulator(
#                 dim_in=latent_dim,
#                 dim_hidden=hidden_features,
#                 num_layers=hidden_layers
#             )
#             # initialize latent input to the modulator network which will also be optimizable
#             self.latent = nn.Parameter(torch.zeros(latent_dim).normal_(0, 1e-2))
#             # append all optimizable parameters in the latent input and modulator network
#             self.opt_latent_parameters.extend([
#                 self.latent,
#                 *self.modulator.parameters(),
#             ])
#
#         # append first layer and all hidden layers
#         self.model = []
#         for i in range(hidden_layers):
#             idx_str = f"{i:4d}_SineLayer"
#             is_first = i == 0
#             omega_0 = first_omega_0 if is_first else hidden_omega_0
#             in_dim = input_dim if is_first else hidden_features
#             self.model.append(
#                 (idx_str, SineLayer(in_dim, hidden_features,
#                                     is_first=is_first, omega_0=omega_0))
#             )
#
#         # append last layer
#         self.model.append(
#             ("LastLayer", nn.Sequential(nn.Linear(hidden_features, output_dim), nn.Tanh()))
#         )
#
#         # ModuleDict makes it easier to get layers by name
#         self.model = nn.ModuleDict(OrderedDict(self.model))
#         # get all Siren optimizable parameters as a list for the Siren Adam Optimizer
#         for l in self.model.values():
#             self.opt_siren_parameters.extend(l.parameters())
#         # if using modulation, initialize its optimizer and save its learning rate
#         if self.has_latent:
#             self.latent_optimizer = optim.Adam(self.opt_latent_parameters, lr=latent_lrate)
#             self.latent_lrate = latent_lrate
#         else:
#             self.latent_optimizer = None
#             self.latent_lrate = None
#
#         self.siren_optimizer = optim.Adam(self.opt_siren_parameters, lr=siren_lrate)
#         self.siren_lrate = siren_lrate
#         self.loss_fn = nn.MSELoss(reduction='none')  # only used for 'step_naive' method
#
#         # set linear LR scheduler
#         self.siren_scheduler, self.latent_scheduler = None, None
#         if scheduler_type == 'step':
#             assert step_size is not None and gamma is not None, "Must specify step size and gamma to use Step LR"
#             self.siren_scheduler = optim.lr_scheduler.StepLR(self.siren_optimizer, step_size=step_size,
#                                                              gamma=gamma)
#             if self.has_latent:
#                 self.latent_scheduler = optim.lr_scheduler.StepLR(self.latent_optimizer, step_size=step_size,
#                                                                   gamma=gamma)
#         elif scheduler_type == 'linear':
#             decay_func_siren = partial(linear_decay, initial_lr=siren_lrate, final_lr=final_siren_lrate,
#                                        total_epochs=num_epochs, last_decay=1e-3)
#             self.siren_scheduler = optim.lr_scheduler.LambdaLR(self.siren_optimizer, lr_lambda=decay_func_siren)
#             if self.has_latent:
#                 decay_func_latent = partial(linear_decay, initial_lr=latent_lrate, final_lr=final_latent_lrate,
#                                             total_epochs=num_epochs, last_decay=1e-2)
#                 self.latent_scheduler = optim.lr_scheduler.LambdaLR(self.latent_optimizer, lr_lambda=decay_func_latent)
#         elif scheduler_type == 'none':
#             pass
#         else:
#             raise ValueError(f"Scheduler type of {scheduler_type} is not recognized")
#
#     def _get_mods(self) -> Tuple[Union[None, Tensor], ...]:
#         # create mods (simply tuple of Nones if not enabled)
#         if self.has_latent:
#             latent_input = self.latent
#             mods = self.modulator(latent_input)
#         else:
#             mods = tuple([None] * self.hidden_layers)
#         return mods
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Simple forward pass of the network
#         :param x: (batches, 3)
#         :return:
#         """
#
#         # get mods
#         mods = self._get_mods()
#
#         hidden_layers = tuple([l for (k, l) in self.model.items() if k.split('_')[-1] == 'SineLayer'])
#         last_layer = self.model['LastLayer']
#         for l, mod in zip(hidden_layers, mods):
#             # pass through sine layer
#             x = l(x)
#
#             # apply mod if feature is enabled
#             if mod is not None:
#                 x *= mod.unsqueeze(0)  # singleton allows mod to be broadcast to all batches
#
#         # apply output layer
#         x = last_layer(x)
#
#         return x
#
#     def forward_with_coords(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#
#         Before the forward pass, clone the input and enable its gradient. Returning this cloned input allows the
#         output of the network to be differentiated w.r.t. the input.
#
#         :param x: (batches, input_dim)
#         :return:
#         """
#         x = x.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
#
#         output = self.forward(x)
#
#         return output, x
#
#     @staticmethod
#     def maximal_lyapunov_fn(traj: Tensor) -> Tensor:
#         """
#         The true maximal Lyapunov function takes value:
#
#         * ∫₀ᴴ ‖x(t;x₀)‖ dt where H = ∞ if the initial state x₀ is stabilizable
#         * ∞ if the initial state x₀ is not stabilizable
#
#         Since it is difficult to calculate this integral, we instead perform a discrete calculation
#         that serves as an estimate of the true maximal Lyapunov function definition.
#         This function returns the value:
#
#         ∑ᵢ₌₀ᴺ ‖x(i⋅Δt; x₀)‖
#
#         where Δt is the discrete time step of our system, and H = Δt⋅N where 1 < N < ∞ or the
#         number of discrete time steps to run the trajectory for.
#
#         :param traj:
#         :return:
#         """
#         traj_norm = torch.linalg.vector_norm(traj, dim=2)
#         lya_discrete = traj_norm.sum(dim=1, keepdim=True)
#
#         return lya_discrete
#
#     @staticmethod
#     def project_to_box_border(x: Tensor, x_L: Tensor, x_U: Tensor) -> Tensor:
#         """
#
#         :param x:       box samples
#         :param x_L:     lower limit box to project onto
#         :param x_U:     upper limit box to project onto
#         :return:
#         """
#
#         # For points inside the box, project to the nearest boundary
#         x_L = x_L.reshape(1, -1).expand(x.shape[0], -1)
#         x_U = x_U.reshape(1, -1).expand(x.shape[0], -1)
#         b = torch.arange(x.shape[0])
#
#         # x = torch.clamp(x, min=x_L, max=x_U)
#         min_l, min_l_idx = torch.min((x - x_L).abs(), dim=1)
#         min_u, min_u_idx = torch.min((x - x_U).abs(), dim=1)
#
#         mask = min_l < min_u
#         min_idx = torch.where(mask, min_l_idx, min_u_idx)
#         border = torch.where(mask, x_L[b, min_l_idx], x_U[b, min_u_idx])
#
#         x[b, min_idx] = border  # perform projection
#
#         return x
#
#     def step(self, x: Tensor, x_b: Tensor, u: Callable) -> Tuple[float, List[float]]:
#         """
#
#         :param x: (batch_size, input_dim)
#         :return:
#         """
#
#         # function to calculate gradients of y w.r.t. x
#         def _gradient(x: Tensor, y: Tensor, grad_outputs=None):
#             if grad_outputs is None:
#                 grad_outputs = torch.ones_like(y)
#             grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
#             return grad
#
#         psi = lambda t: torch.linalg.vector_norm(t, dim=1, keepdim=True)
#
#         # Get points on the border of the region of interest, ∂R₂ where R₂ = {αx : x ∈ R₁},
#         # R₁ ⊆ 𝒟, and 𝒟 is the region from which we are drawing samples.
#         # x_b = ZubovNetwork.project_to_box_border(self.alpha*x, self.lb, self.ub)
#
#         with torch.no_grad():
#             # in case dynamical system is described as another NN, we do not want to update its parameter
#             # fx = self.dynamical_system(x)  # return the next state
#             traj = self.dynamical_system(x, u, self.num_steps)  # return the trajectory
#             fx = traj[:, 0, :]  # get the next state
#             max_v = ZubovNetwork.maximal_lyapunov_fn(traj)
#             psi_output = psi(x)
#
#         # zero the gradients
#         self.siren_optimizer.zero_grad()
#         if self.has_latent:
#             self.latent_optimizer.zero_grad()
#
#         yhat, coords = self.forward_with_coords(x)  # zubov output
#
#         grad_output = _gradient(coords, yhat)  # grad of zubov w.r.t. x
#
#         # Loss Z ensures that at the equilibrium, the output of W(x) is 0.
#         loss_z = (self.forward(torch.zeros((1, self.input_dim))) ** 2)
#         # Loss R ensures that W(x) = tanh(a*V(x)) where V(x) in this implementation is
#         # an approximation of the defined Maximal Lyapunov function
#         loss_r = (yhat - torch.tanh(self.alpha * max_v)) ** 2
#         # Loss P is the derivative condition: ∂ₓW(x)ᵀf(x) = a(1−W(x))(1+W(x))Φ(x)
#         # where Φ(x) := ‖x‖
#         loss_p1 = torch.einsum('bi,bi->b', grad_output, fx).unsqueeze(1)
#         loss_p2 = self.alpha * (1 - yhat) * (1 + yhat) * psi_output
#         loss_p = (loss_p1 + loss_p2) ** 2
#         # loss_b = self.lambda_b * (self.forward(x_b) - 1).abs()
#         loss_b = 0.
#         loss = self.c1 * loss_z + self.c2 * loss_r + self.c3 * loss_p + loss_b
#         total_loss = loss.mean()
#
#         # perform backward gradient calculations
#         total_loss.backward()
#
#         # perform gradient clipping
#         # typically recommended for stable training
#         if self.clip_gradient_norm is not None:
#             nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_gradient_norm)
#
#         # perform gradient clipping
#         # typically recommended for stable training
#         if self.clip_gradient_norm is not None:
#             torch.nn.utils.clip_grad_norm_(self.opt_siren_parameters, max_norm=self.clip_gradient_norm)
#             if self.has_latent:
#                 torch.nn.utils.clip_grad_norm_(self.opt_latent_parameters, max_norm=self.clip_gradient_norm)
#
#         # optimize all parameters
#         self.siren_optimizer.step()
#         if self.has_latent:
#             self.latent_optimizer.step()
#
#         rest_losses = [t.mean().item() if isinstance(t, Tensor) else t for t in [loss_z, loss_r, loss_p, loss_b]]
#         return total_loss.item(), *rest_losses
#
#     def scheduler_step(self) -> Tuple[float, Optional[float]]:
#         """
#         Steps the siren and latent schedulers if they are being used.
#         In addition, returns the siren learning rate (should always exist) and latent learning rate (optionally exists)
#         before the scheduling step.
#         :return:
#         """
#
#         # step with siren scheduler
#         if self.siren_scheduler is not None:
#             self.siren_scheduler.step()
#             siren_lr = self.siren_scheduler.get_last_lr()[0]
#         else:
#             siren_lr = self.siren_lrate
#
#         # step with latent scheduler
#         if self.has_latent and self.latent_scheduler is not None:
#             self.latent_scheduler.step()
#             latent_lr = self.latent_scheduler.get_last_lr()[0]
#         else:
#             latent_lr = self.latent_lrate
#
#         return siren_lr, latent_lr


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