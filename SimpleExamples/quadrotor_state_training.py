from typing import Union, Tuple, Optional, List, Dict
import os
import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from new_quadrotor_dynamics import QuadrotorDynamics
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch import Tensor
import wandb
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.train_utils as train_utils
import scipy

device = torch.device("cpu")
dtype = torch.float

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

def compute_lqr(quadrotor_tracking_continous: QuadrotorDynamics):
    x_equilibrium = quadrotor_tracking_continous.x_equilibrium.to(device)
    u_equilibrium = quadrotor_tracking_continous.u_equilibrium.to(device)
    t_A, t_B = quadrotor_tracking_continous.linearized_dynamics(
       x_equilibrium, u_equilibrium
    )
    A = to_numpy(t_A)
    B = to_numpy(t_B)
    # check to see that the system is controllable
    ctrl_matrix = get_controllability_matrix(A, B)
    ctrl_rank = np.linalg.matrix_rank(ctrl_matrix)
    can_ctrl = ctrl_rank >= A.shape[1]
    assert can_ctrl, f"This system is not controllable."

    Q = np.eye(quadrotor_tracking_continous.nx)
    Q[:3] *= 400 # increases cost for positions being away from equilibrium
    R = np.eye(quadrotor_tracking_continous.nu)
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ S)
    return K, S

def get_controllability_matrix(a: ndarray, b: ndarray) -> ndarray:
    """
    For an LTI system, return its controllability matrix.
    :param a: Linear state equations w.r.t. current state.
    :param b: Linear state equations w.r.t. input.
    :return:
    """
    n = a.shape[1]  # state dimension
    m = b.shape[1]  # control dimension

    # Use Cayley-Hamilton theorem to create a (n, nxm) controllability matrix whose rank tells us
    # how many states in the system are controllable.
    ctrl_matrix = b
    for i in range(1, n):
        ctrl_matrix = np.hstack((ctrl_matrix, np.linalg.matrix_power(a, i)@b))
    assert ctrl_matrix.shape == (n, n*m), f"Controllability matrix does not have the proper shape of ({(n, n*m)})"
    return ctrl_matrix

def approximate_lqr(
    quadrotor_tracking_continous: QuadrotorDynamics,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    lower_limit: Tensor,
    upper_limit: Tensor,
    logger,
):
    K, S = compute_lqr(quadrotor_tracking_continous)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    range_limit = (upper_limit - lower_limit).reshape(1, -1)
    lower_limit = lower_limit.reshape(1, -1)
    upper_limit = upper_limit.reshape(1, -1)
    x = (torch.rand((100000, quadrotor_tracking_continous.nx), dtype=dtype, device=device) * range_limit) + lower_limit
    # V = torch.sum(x * (x @ S_torch), axis=1, keepdim=True)
    x_bar = x - quadrotor_tracking_continous.x_equilibrium.reshape(1, -1)
    batches = len(x_bar)
    V = torch.einsum("bn,nm,bm->b", x_bar, S_torch, x_bar).reshape(batches, 1)
    u = quadrotor_tracking_continous.u_equilibrium.reshape(1, -1) + torch.einsum("mn,bn->bm", -K_torch, x_bar)

    def approximate(system, system_input, target, lr, max_iter):
        optimizer = torch.optim.Adam(system.parameters(), lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            output = torch.nn.MSELoss()(system.forward(system_input), target)
            logger.info(f"iter {i}, loss {output.item()}")
            output.backward()
            optimizer.step()

    print("Approximating the LQR controller")
    approximate(controller, x, u, 0.01, 500)
    print("Approximating the Lyapunov function from LQR")
    approximate(lyapunov_nn, x_bar, V, 0.5, 10000)


def plot_V_heatmap(V, lower_limit, upper_limit, rho):
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 1000, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 1000, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    with torch.no_grad():
        V_val = V.forward(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)
    V_val = V_val.cpu()
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val)
    ax.contour(grid_x, grid_y, V_val, [rho], colors="red")
    ax.set_xlim(lower_limit[0], upper_limit[0])
    ax.set_ylim(lower_limit[1], upper_limit[1])
    ax.set_xlabel("distance")
    ax.set_ylabel("angle")
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


@hydra.main(config_path="./config",
            config_name="quadrotor_state_training.yaml")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    dt = cfg.model.dt
    quadrotor_tracking_continous = QuadrotorDynamics()
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        quadrotor_tracking_continous,
        dt=dt
    )
    # dynamics = dynamical_system.QuadrotorSystem(
    #     quadrotor_tracking_continous,
    #     dt=dt)

    # update the equilibrium point of the system
    quadrotor_tracking_continous.x_equilibrium = torch.tensor([
        10., 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    quadrotor_tracking_continous.u_equilibrium = torch.tensor([14475.809152959684] * 4)
   
    controller = controllers.NeuralNetworkController(
        nlayer=5,
        in_dim=quadrotor_tracking_continous.nx,
        out_dim=quadrotor_tracking_continous.nu,
        hidden_dim=64,
        clip_output="clamp",
        u_lo=torch.tensor([0.]*4),
        u_up=torch.tensor([18000]*4),  # reported upper limit of crazyflie 2.0 drone
        x_equilibrium=quadrotor_tracking_continous.x_equilibrium,
        u_equilibrium=quadrotor_tracking_continous.u_equilibrium,
    )
    controller.eval()

    absolute_output = True
    if cfg.model.lyapunov.quadratic:
        _, S = compute_lqr(quadrotor_tracking_continous)
        S_torch = torch.from_numpy(S).type(dtype).to(device)
        R = torch.linalg.cholesky(S_torch)
        lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
            goal_state=quadrotor_tracking_continous.x_equilibrium.to(device),
            x_dim=quadrotor_tracking_continous.nx,
            R_rows=quadrotor_tracking_continous.nx,
            eps=0.01,
            R=R,
        )
    else:
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=quadrotor_tracking_continous.x_equilibrium,
            hidden_widths=cfg.model.lyapunov.hidden_widths,
            x_dim=12,
            R_rows=3,
            absolute_output=absolute_output,
            eps=0.01,
            activation=nn.LeakyReLU,
            V_psd_form=cfg.model.V_psd_form,
        )
    lyapunov_nn.eval()

    kappa = cfg.model.kappa
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=0,  # will be updated in the training for-loop
        box_up=0,  # will be updated in the training for-loop
        rho_multiplier=1,
        kappa=kappa,
        hard_max=cfg.train.hard_max,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([50]*(quadrotor_tracking_continous.nx), device=device)
    logger = logging.getLogger(__name__)
    if cfg.approximate_lqr:
        # Get the upper limit for the first limit_scale and get the NN controller and lya function to approximate the
        # LQR controller and lya function.
        limit_scale = cfg.model.limit_scale[0]
        lya_limit_epsilon = limit_scale * torch.tensor(cfg.model.limit, device=device)
        lya_lower_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) - lya_limit_epsilon
        lya_upper_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) + lya_limit_epsilon
        approximate_lqr(
            quadrotor_tracking_continous, controller, lyapunov_nn, lya_lower_limit, lya_upper_limit, logger
        )
        print("Done approximating LQR, saving the model...")
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict()},
            os.path.join(os.getcwd(), "lyaloss_lqr.pth"),
        )

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = os.path.join(
            os.path.dirname(__file__), "../", cfg.model.load_lyaloss
        )
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    # if the output is not absolute, then we should incur loss wherever the Lyapunov function is negative as it
    # it should be PSD
    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(quadrotor_tracking_continous.nx, device=device)
        )

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )
        # wandb.config.update(cfg)

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa

    if cfg.train.derivative_x_buffer_path is not None:
        derivative_x_buffer = torch.load(cfg.train.derivative_x_buffer_path)
    else:
        derivative_x_buffer = None

    if cfg.train.train_lyaloss:
        # start of Controller and Lyapunov network training

        # we will iteratively increase the input box in order to train Lyapunov functions that verify larger level-sets
        for n in range(len(cfg.model.limit_scale)):

            # get the input box for this iteration
            limit_scale = cfg.model.limit_scale[n]
            limit_epsilon = limit_scale * torch.tensor(cfg.model.limit, device=device)
            lower_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) - limit_epsilon
            upper_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) + limit_epsilon

            # Constructs the Lyapunov derivative loss which ensures that states covered by the rho level-set
            # indeed are Lyapunov stable
            derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
                dynamics,
                controller,
                lyapunov_nn,
                box_lo=lower_limit,
                box_up=upper_limit,
                rho_multiplier=cfg.model.rho_multiplier[n],
                kappa=kappa,
                hard_max=cfg.train.hard_max,
            )
 
            # saves the model parameters during each iteration
            if save_lyaloss:
                save_lyaloss_path = os.path.join(
                    os.getcwd(), f"lyaloss_{limit_scale}.pth"
                )
            else:
                save_lyaloss_path = None

            # construct a Tensor of states that we want to ensure are Lyapunov stable
            candidate_roa_states = limit_scale * torch.tensor(
                cfg.loss.candidate_roa_states,
                device=device,
            )

            # start training the controller and Lyapunov function
            train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=None,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                grid_size=grid_size,
                learning_rate=cfg.train.learning_rate,
                weight_decay=0.0,
                max_iter=cfg.train.max_iter,
                enable_wandb=cfg.train.wandb.enabled,
                derivative_ibp_ratio=cfg.loss.ibp_ratio_derivative,
                derivative_sample_ratio=cfg.loss.sample_ratio_derivative,
                positivity_ibp_ratio=cfg.loss.ibp_ratio_positivity,
                positivity_sample_ratio=cfg.loss.sample_ratio_positivity,
                save_best_model=save_lyaloss_path,
                pgd_steps=cfg.train.pgd_steps,
                buffer_size=cfg.train.buffer_size,
                batch_size=cfg.train.batch_size,
                epochs=cfg.train.epochs,
                samples_per_iter=cfg.train.samples_per_iter,
                l1_reg=cfg.loss.l1_reg,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
                Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight,
                derivative_x_buffer=derivative_x_buffer,
                logger=logger,
                always_candidate_roa_regularizer=cfg.loss.always_candidate_roa_regularizer,
            )

        # save the final models
        torch.save(
            {
                "state_dict": lyapunov_nn.state_dict(),
                "rho": derivative_lyaloss.get_rho(),
            },
            os.path.join(os.getcwd(), "lyapunov_nn.pth"),
        )
    else:
        # we do not want to train any network but rather the maximum (or minimum) value of a previously trained
        # Lyapunov network at inputs that lie on the border of the input box
        limit_scale = cfg.model.limit_scale[-1]
        limit_epsilon = limit_scale * torch.tensor(cfg.model.limit, device=device)
        lower_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) - limit_epsilon
        upper_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) + limit_epsilon
        derivative_lyaloss.x_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            eps=limit_epsilon,
            steps=100,
            direction="minimize",
        )

    # we now check for counter-examples to see if the Lyapunov derivative constraint is violated anywhere in the
    # input box
    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=cfg.model.rho_multiplier[-1],
        kappa=0.0,
        hard_max=True,
    )


    pgd_verifier_find_counterexamples = False
    counterexamples_check = torch.zeros((0, 12), device=device)
    pgd_steps = 100
    for seed in range(pgd_steps):
        train_utils.set_seed(seed)

        # If true, then we want to check that the Lyapunov derivative constraint is satisfied only within the rho
        # level-set of the Lyapunov function. Otherwise, we want to check that the constraint is satisfied at ALL
        # points inside the input box
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                eps=limit_epsilon,
                steps=pgd_steps,
                direction="minimize",
            )
            if derivative_lyaloss.x_boundary is not None:
                derivative_lyaloss_check.x_boundary = torch.cat(
                    (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
                )

        # randomly sample the input box
        limit_scale = cfg.model.limit_scale[-1]
        limit_epsilon = limit_scale * torch.tensor(cfg.model.limit, device=device)
        lower_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) - limit_epsilon
        upper_limit = quadrotor_tracking_continous.x_equilibrium.to(device=device) + limit_epsilon
        limit_range = (upper_limit - lower_limit).reshape(1, -1)
        x_check_start = (
            (
                torch.rand((50000, 12), device=device)
                - torch.full((12,), 0.5, device=device)
            )
            * limit_range
            + lower_limit.reshape(1, -1)
        )

        # run pgd attacks to find and return counter examples
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit_epsilon,
            steps=cfg.pgd_verifier_steps,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()

        # if no counter-examples were found, then the loss 'adv_output' will be 0, otherwise it will be positive
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        counterexamples_check = torch.cat(
            (counterexamples_check, adv_x[adv_output.squeeze(1) > 0]), dim=0
        )
        if max_adv_violation > 0:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)

    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )
    # save the counter-examples in case we want to further evaluate these points
    if counterexamples_check.shape[0] > 0:
        torch.save(
            counterexamples_check,
            os.path.join(os.getcwd(), "counterexamples_check.pth"),
        )

    # Choose random points in the input box to visualize their trajectory. If the Lyapunov function can be verified,
    # then any points inside the rho level-set are guaranteed to converge to equilibrium.
    x0 = (
            torch.rand((40, 12), device=device)
            * limit_range
            + lower_limit.reshape(1, -1)
          )
    x_traj, V_traj = models.simulate(derivative_lyaloss, 500, x0)
    plt.plot(torch.stack(V_traj).cpu().detach().squeeze().numpy())
    plt.savefig(os.path.join(os.getcwd(), "Vtraj_roa.png"))

    # plots a heat-map of the level-sets of the Lyapunov function where the 0 level-set is the equilibrium
    rho = derivative_lyaloss.get_rho().item()
    print("rho = ", rho)
    fig = plt.figure()
    train_utils.plot_V_heatmap(
        fig,
        lyapunov_nn,
        rho,
        lower_limit,
        upper_limit,
        quadrotor_tracking_continous.nx,
        derivative_lyaloss.x_boundary,
    )
    fig.show()
    plt.savefig(os.path.join(os.getcwd(), "V_roa.png"))

if __name__ == "__main__":
    main()
