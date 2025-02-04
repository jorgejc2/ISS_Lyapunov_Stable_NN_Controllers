import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

# FIXME: Make this as a clean import, but for now we hardcode our PyTorch dynamics
import torch
import torch.nn as nn
from torch import Tensor
from numpy import ndarray
from typing import Optional, Union, Tuple
from enum import Enum

class QuadrotorDynamics:

    def __init__(self,
                 m: float = 0.027, j_x: float = 2.3951e-5, j_y: float = 2.3951e-5, j_z: float = 3.2347e-5,
                 arm_length: float = 0.0397, kf: float = 3.16e-10, km: float = 7.94e-12, g: float = 9.81,
                 thrust2weight: float = 2.25, max_speed_kmh: float = 30., gnd_eff_coeff: float = 11.36859,
                 prop_radius: float = 2.31348e-2, drag_coeff_xy: float = 9.1785e-7, drag_coeff_z: float = 10.311e-7,
                 dw_coeff_1: float = 2267.18, dw_coeff_2: float = 0.16, dw_coeff_3: float = -0.11
                 ):
        """
        Initializes quadrotor model with 6 degrees of freedom. Dynamics faithfully follow those described by Daniel
        Mellinger in "Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors". Default
        parameters were selected by Panerati et. al. in "Leawrning to Fly - a Gym Environment with PyBullet Physics
        for Reinforcement Learning of Multi-agent Quadcopter Control".
        :param m:               Mass of the quadrotor (kg)
        :param j_x:             Moment of inertia around x-axis (kg·m²)
        :param j_y:             Moment of inertia around y-axis (kg·m²)
        :param j_z:             Moment of inertia around z-axis (kg·m²)
        :param arm_length:      Distance from the center to a propeller (m)
        :param kf:              Thrust coefficient
        :param km:              Torque coefficient
        :param g:               Gravitational acceleration (m/s²)
        :param thrust2weight:   Thrust-to-weight ratio
        :param max_speed_kmh:   Maximum speed (km/h)
        :param gnd_eff_coeff:   Ground effect coefficient
        :param prop_radius:     Propeller radius (m)
        :param drag_coeff_xy:   Drag coefficient in XY plane
        :param drag_coeff_z:    Drag coefficient in Z direction
        :param dw_coeff_1:      Coefficients for downwash effect
        :param dw_coeff_2:      Coefficients for downwash effect
        :param dw_coeff_3:      Coefficients for downwash effect
        """
        # Dimensions of state and control input
        self.nx = 12  # Number of state dimensions
        self.nu = 4   # Number of control inputs
        self.m = m  # Mass of the quadrotor (kg)
        self.J_x = j_x  # Moment of inertia around x-axis (kg·m²)
        self.J_y = j_y  # Moment of inertia around y-axis (kg·m²)
        self.J_z = j_z  # Moment of inertia around z-axis (kg·m²)
        self.J = torch.diag(torch.tensor[self.J_x, self.J_y, self.J_z])  # Moment of inertia matrix
        self.J_INV = torch.linalg.inv(self.J)  # Inverse of the moment of inertia matrix
        self.arm_length = arm_length  # Distance from the center to a propeller (m)
        self.kf = kf  # Thrust coefficient
        self.km = km  # Torque coefficient
        self.g = g  # Gravitational acceleration (m/s²)
        # Additional properties
        self.thrust2weight = thrust2weight  # Thrust-to-weight ratio
        self.max_speed_kmh = max_speed_kmh  # Maximum speed (km/h)
        self.gnd_eff_coeff = gnd_eff_coeff  # Ground effect coefficient
        self.prop_radius = prop_radius  # Propeller radius (m)
        self.drag_coeff_xy = drag_coeff_xy  # Drag coefficient in XY plane
        self.drag_coeff_z = drag_coeff_z  # Drag coefficient in Z direction
        self.dw_coeff_1 = dw_coeff_1  # Coefficients for downwash effect
        self.dw_coeff_2 = dw_coeff_2
        self.dw_coeff_3 = dw_coeff_3

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """
        A forward pass of the model dynamics. Dynamics faithfully follow those described by Daniel Mellinger in
        "Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors".
        x: state (batch, 12); 
        u: controller input (batch, 4).
        """
        # States
        batch = x.shape[0]
        pos_x, pos_y, pos_z = x[:, 0], x[:, 1], x[:, 2]  # Positions
        roll, pitch, yaw = x[:, 3], x[:, 4], x[:, 5]  # Angles (orientation)
        vel_x, vel_y, vel_z = x[:, 6], x[:, 7], x[:, 8]  # Velocities
        ang_vel_x, ang_vel_y, ang_vel_z = x[:, 9], x[:, 10], x[:, 11]  # Angular velocities

        # Control inputs (motor RPMs)
        rpm_motor_1, rpm_motor_2, rpm_motor_3, rpm_motor_4 = (
            u[:, 0], u[:, 1], u[:, 2], u[:, 3])

        forces = (u**2) * self.kf # shape (batch, nu)
        thrust = torch.zeros(batch, 3)
        thrust[:, 2] = forces.sum(1)
        
        quat = torch.stack((roll,pitch,yaw), dim=1) # shape (batch, 3)
        rotation = self.compute_rotation_matrix(quat) # shape (batch, 3, 3)
        # perform batch matrix multiplication
        thrust_world_frame = torch.einsum('bmn,bn->bm', rotation, thrust)
        # b = batch, m = 3, n = 3 (even though m=n, we use different letters to eliminate ambiguity)
        # add singleton dimension by unsqueezing so that the subtraction is done on every batch
        force_world_frame = thrust_world_frame - torch.tensor([0, 0, self.g]).unsqueeze(0)
        acc = force_world_frame / self.m

        # Angular Dynamics
        ang_vels = torch.stack((ang_vel_x, ang_vel_y, ang_vel_z), dim=1) # shape(batch, 3)
        z_torque = (u**2) * self.km
        x_torque = (forces[:, 1] - forces[:, 3]) * self.arm_length
        y_torque = (-forces[:,0] + forces[:, 2]) * self.arm_length
        torques = torch.stack((x_torque, y_torque, z_torque), dim=1) # shape (batch, 3)
        
        torques = torques - torch.cross(ang_vels, torch.einsum('mn,bn->bm', self.J, ang_vels), dim=1)
        angular_acc = torch.einsum('bmn,bn->bm', self.J_INV, torques)
        # b = batch, m = 3, n = 3

        # Kinematic equations
        dx = torch.zeros_like(x)
        dx[:, 0:3] = x[:, 3:6]  # Position derivatives (velocity)
        dx[:, 3:6] = x[:, 6:9]  # Angular position derivatives (angular velocity)
        dx[:, 6:9] = acc  # Velocity derivatives (acceleration)
        dx[:, 9:12] = angular_acc  # Angular velocity derivatives (angular acceleration)
        return dx

    def linearized_dynamics(self, x, u):
        # FIXME: This linearized dynamics are not correct. Ideally, the linearization should be cross-checked
        # with a symbolic tool that Matlab and SymPy both provide.
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx))
        B = torch.zeros((batch_size, self.nx, self.nu))
        
        # Position-velocity relationships
        A[:, 0, 3] = 1  # dx1/dx4
        A[:, 1, 4] = 1  # dx2/dx5
        A[:, 2, 5] = 1  # dx3/dx6

        # Linearized velocity relationships
        A[:, 3, 7] = -self.g * torch.sin(x[:, 7])  # dx4/dx8 (pitch affects forward acceleration)
        A[:, 4, 6] = self.g * torch.cos(x[:, 7]) * torch.sin(x[:, 6])  # dx5/dx7 (roll affects lateral acceleration)
        A[:, 5, 6] = -self.g * torch.cos(x[:, 7]) * torch.cos(x[:, 6])  # dx6/dx7 (altitude affected by roll and pitch)

        # Angular velocity-rotation relationships
        A[:, 6, 9] = 1   # d(theta)/d(omega_x)
        A[:, 7, 10] = 1  # d(phi)/d(omega_y)
        A[:, 8, 11] = 1  # d(psi)/d(omega_z)
       
        B[:, 5, 0] = -1 / self.m  # Effect of thrust on vertical acceleration
        B[:, 9, 1] = 1 / self.J_x  # Control effect of u2 (roll torque) on omega_x
        B[:, 10, 2] = 1 / self.J_y  # Control effect of u3 (pitch torque) on omega_y

        return A.to(device), B.to(device)

    ## static methods ##
    @staticmethod
    def compute_rotation_matrix(quat: Tensor) -> Tensor:
        """
        Compute the rotation matrix from roll, pitch, and yaw angles.
        :param quat:    Tensor containing the roll, pitch, and yaw angles for all batches
        :return:        Rotation matrix using these angles to transform from body to world frame
        """
        batch = quat.shape[0]
        roll, pitch, yaw = quat[:, 0], quat[:, 1], quat[:, 2]  # unpack angles

        # precompute trigonometric results
        c_roll, s_roll = torch.cos(roll), torch.sin(roll)
        c_pitch, s_pitch = torch.cos(pitch), torch.sin(pitch)
        c_yaw, s_yaw = torch.cos(yaw), torch.sin(yaw)

        # initialize and fill rotation matrix
        R = torch.zeros((batch, 3, 3), device=roll.device)
        R[:, 0, 0] = c_yaw * c_pitch
        R[:, 0, 1] = c_yaw * s_pitch * s_roll - s_yaw * c_roll
        R[:, 0, 2] = c_yaw * s_pitch * c_roll + s_yaw * s_roll
        R[:, 1, 0] = s_yaw * c_pitch
        R[:, 1, 1] = s_yaw * s_pitch * s_roll + c_yaw * c_roll
        R[:, 1, 2] = s_yaw * s_pitch * c_roll - c_yaw * s_roll
        R[:, 2, 0] = -s_pitch
        R[:, 2, 1] = c_pitch * s_roll
        R[:, 2, 2] = c_pitch * c_roll
        return R

    @staticmethod
    def integrateQ(quat: Tensor, omega: Tensor, time_step: float) -> Tensor:
        """
        Performs careful integration to update the quaternion values (roll, pitch, yaw) from their rates (angular
        velocity). For sufficiently small angle rates, the quaternion is kept constant for stability.
        :param quat:      Tensor containing roll, pitch, and yaw angles for all batches
        :param omega:     Tensor containing the angular velocities (roll rate, pitch rate, and yaw rate) for all batches
        :param time_step: Discrete time step parameter
        :return:          Updated quaternion values
        """
        next_quat = quat.clone()
        omega_norm = torch.linalg.norm(omega, dim=1)
        p, q, r = omega[:, 0], omega[:, 1], omega[:, 2]  # unpack angular rates
        # mask to only update batches whose quaternion rates are not significantly small
        update_mask = torch.logical_not(torch.isclose(omega_norm, torch.zeros_like(omega_norm)))
        num_update = update_mask.sum().item()
        if num_update == 0:
            # no batches to update
            return next_quat

        # filter out angles and norms that will not be updated
        quat = quat[update_mask]
        p = p[update_mask]
        q = q[update_mask]
        r = r[update_mask]
        omega_norm = omega_norm[update_mask]

        # put angles into skew matrix form (so that we can do matrix multiplication instead of cross multiplication)
        batch_zeros = torch.zeros_like(r)
        lambda_ = torch.stack([
            torch.stack([batch_zeros, r, -q, p], dim=1),
            torch.stack([-r, batch_zeros, p, q], dim=1),
            torch.stack([q, -p, batch_zeros, r], dim=1),
            torch.stack([-p, -q, -r, batch_zeros], dim=1)
        ], dim=1) * 0.5  # shape (batch, 4, 4)
        theta = omega_norm * time_step / 2
        # Intermediate calculation for calculating the next quaternion; Reshaping Tensors is another way to add singleton
        # dimensions while also explicitly listing the shapes. Singleton dimensions allow for broadcasting, i.e. the
        # torch.eye(4).reshape(1,4,4)*torch.cos(theta).reshape(num_update,1,1) term produces a Tensor that has shape
        # (num_update, 4, 4) where each 4x4 identity matrix is multiplied by its corresponding cos(theta) scalar value.
        inter_quat = torch.eye(4).reshape(1,4,4)*torch.cos(theta).reshape(num_update,1,1) + (2*torch.sin(theta)/omega_norm).reshape(num_update,1,1)*lambda_
        # Formalize the next quaternion for the values that should be updated
        next_quat[update_mask, :] = torch.einsum('bmn,bn->bn', inter_quat, quat)
        return next_quat

    ## properties ##
    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))

class DiscreteTimeSystem(nn.Module):
    """
    Defines the interface for the discrete dynamical system.
    """

    def __init__(self, nx, nu, *args, **kwargs):
        super(DiscreteTimeSystem, self).__init__(*args, **kwargs)
        self.nx = nx
        self.nu = nu
        pass

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def x_equilibrium(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def u_equilibrium(self) -> torch.Tensor:
        raise NotImplementedError


class IntegrationMethod(Enum):
    ExplicitEuler = 1
    MidPoint = 2

class SecondOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    second-order dynamical system in time.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        position_integration: IntegrationMethod = IntegrationMethod.MidPoint,
        velocity_integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          qddot = f(x, u) where x = [q, qdot].
        """
        super(SecondOrderDiscreteTimeSystem, self).__init__(
            continuous_time_system.nx, continuous_time_system.nu
        )
        assert callable(getattr(continuous_time_system, "forward"))
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.nq = int(self.nx / 2)
        self.dt = dt
        self.velocity_integration = velocity_integration
        self.position_integration = position_integration
        self.continuous_time_system = continuous_time_system
        self.Ix = torch.eye(self.nx)

    def forward(self, x, u):
        """
        Compute x_next for a batch of x and u
        """
        assert x.shape[0] == u.shape[0]
        qddot = self.continuous_time_system.forward(x, u)
        if self.velocity_integration == IntegrationMethod.ExplicitEuler:
            qdot_next = x[:, self.nq :] + qddot * self.dt
        else:
            raise NotImplementedError
        if self.position_integration == IntegrationMethod.MidPoint:
            q_next = x[:, : self.nq] + (qdot_next + x[:, self.nq :]) / 2 * self.dt
        elif self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next = x[:, : self.nq] + x[:, self.nq :] * self.dt
        else:
            raise NotImplementedError
        return torch.cat((q_next, qdot_next), dim=1)

    def linearized_dynamics(self, x, u):
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        Ad = self.dt * Ac + self.Ix.to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd

    # def output_feedback_linearized_lyapunov(self, K, L):
    #     """
    #     Given the control gain K and observer gain L, solve the discrete-time Lyapunov equation
    #     for the closed-loop system with the states and controls at equilibrium.
    #     The linearized dynamics are computed from the continuous-time system with Explicit Euler method.
    #     """
    #     x0 = self.x_equilibrium.unsqueeze(0)
    #     Ad, Bd = self.continuous_time_system.linearized_dynamics(
    #         x0, self.u_equilibrium.unsqueeze(0)
    #     )
    #     Ad = Ad.squeeze().detach().numpy()
    #     Bd = Bd.squeeze().detach().numpy()
    #     C = (
    #         self.continuous_time_system.linearized_observation(x0)
    #         .squeeze()
    #         .detach()
    #         .numpy()
    #     )
    #     Acl = np.vstack(
    #         (
    #             np.hstack((Ad + Bd @ K, -Bd @ K)),
    #             np.hstack((np.zeros([self.nx, self.nx]), Ad - L @ C)),
    #         )
    #     )
    #     Acl[np.abs(Acl) <= 1e-6] = 0
    #     S = control.dlyap(Acl, np.eye(2 * self.nx))
    #     return S

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


class QuadrotorSystem(SecondOrderDiscreteTimeSystem):

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        pre_mask = np.zeros(12)
        pre_mask[3:6] = 1
        self._a_mask = np.argwhere(pre_mask == 1)[0].tolist()  # angular mask
        self._r_mask = np.argwhere(pre_mask == 1)[0].tolist()  # rest mask
        
    """
    q = [position_x, position_y, position_z, roll, pitch, yaw]
    qdot = [velocity_x, velocity_y, velocity_z, roll rate, pitch rate, yaw rate]
    x = [q, qdot]
    qddot = f(x,u) = [velocity_x, velocity_y, velocity_z, roll rate, pitch rate, yaw rate, acceleration_x, acceleration_y, acceleration_z, roll acceleration, pitch acceleration, yaw acceleration]
    q, qdot are vectors of length 6. qddot and x is a vector of length 12. qddot is essentially the derivative of x
    Roll, pitch, and yaw should be updated using the integrateQdot method. Everything else may be updated using Euler integration.
    """
    def integrateQdot(self, quat_vel, omega, dt):

        quat_vel = torch.tensor(quat_vel, dtype=torch.float32)
        omega = torch.tensor(omega, dtype=torch.float32)
        
        omega_norm = torch.linalg.norm(omega)
        pdot, qdot, rdot = omega
        
        if torch.isclose(omega_norm, torch.tensor(0.0)):
            return quat_vel
        
        lambda_ = torch.tensor([
            [0, rdot, -qdot, pdot],
            [-rdot, 0, pdot, qdot],
            [qdot, -pdot, 0, rdot],
            [-pdot, -qdot, -rdot, 0]
        ], dtype=torch.float32) * 0.5
        
        theta = omega_norm * dt / 2
        
        # Calculate updated quaternion
        quat_vel = (torch.eye(4, dtype=torch.float32) * torch.cos(theta) +
                    2 / omega_norm * lambda_ * torch.sin(theta)) @ quat_vel
        
        return quat_vel

    def forward(self, x, u):
        """
        Compute x_next for a batch of x and u
        x: 12 dimensional [q, qdot]
        u: 4 dimensional
        angular_acc: 3 dimensional [roll_acc, pitch_acc, yaw_acc]
        """   
        
        assert x.shape[0] == u.shape[0]

        a_mask = self._a_mask
        r_mask = self._r_mask
        quat = x[:, a_mask]  # Angles (orientation)
        qddot = self.continuous_time_system.forard(x, u)
        angular_vel = qddot[:, a_mask]
        new_quat = QuadrotorDynamics.integrateQ(quat, angular_vel, self.dt)
        x_upper = x[:, self.nq:]  # current velocities
        x_lower = x[:, :self.nq]  # current positions

        q_next, qdot_next = torch.zeros_like(x[:, : self.nq]), torch.zeros_like(x[:, : self.nq])
        q_next[:, a_mask] = new_quat

        # update the velocities (qdot_next)
        if self.velocity_integration == IntegrationMethod.ExplicitEuler:
            qdot_next = x_upper + qddot * self.dt
        else:
            raise NotImplementedError
        
        # # FIXME: Example template, remove later
        # if self.position_integration == IntegrationMethod.MidPoint:
        #     q_next = x[:, : self.nq] + (qdot_next + x[:, self.nq :]) / 2 * self.dt
        # elif self.position_integration == IntegrationMethod.ExplicitEuler:
        #     q_next = x[:, : self.nq] + x[:, self.nq :] * self.dt
        # else:
        #     raise NotImplementedError
        
        # update the positions (q_next)
        x_upper_rest = x_upper[:, r_mask]
        x_lower_rest = x_lower[:, r_mask]
        if self.position_integration == IntegrationMethod.MidPoint:
            qdot_next_rest = qdot_next[:, r_mask]
            q_next_rest = x_lower_rest + (qdot_next_rest + x_upper_rest) / 2 * self.dt
            q_next[:, r_mask] = q_next_rest
        elif self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next_rest = x_lower_rest + x_upper_rest * self.dt
            q_next[:, r_mask] = q_next_rest
        else:
            raise NotImplementedError
        
        return torch.cat((q_next, qdot_next), dim=1)


class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results'
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.PYB_STEPS_PER_CTRL != 0:
                print("[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/"), exist_ok=True)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
    
    ################################################################################

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    ################################################################################

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info
    
    ################################################################################
    
    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.PYB_TIMESTEP, self.PYB_FREQ, (self.step_counter*self.PYB_TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0]*self.RAD2DEG, self.rpy[i, 1]*self.RAD2DEG, self.rpy[i, 2]*self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
    
    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.join(self.OUTPUT_FOLDER, "video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".mp4"),
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), '')
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################

    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool=True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray 
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :]+np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int=0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(os.path.join(path,"frame_"+str(frame_num)+".png"))
        elif img_type == ImageType.DEP:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(os.path.join(path,"frame_"+str(frame_num)+".png"))

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix 
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                if np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j+i+1] = adjacency_mat[j+i+1, i] = 1
        return adjacency_mat
    
    ################################################################################
    
    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    ################################################################################

    def _groundEffect(self,
                      rpm,
                      nth_drone
                      ):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = p.getLinkStates(self.DRONE_IDS[nth_drone],
                                        linkIndices=[0, 1, 2, 3, 4],
                                        computeLinkVelocity=1,
                                        computeForwardKinematics=1,
                                        physicsClientId=self.CLIENT
                                        )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array([link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS/(4 * prop_heights))**2
        if np.abs(self.rpy[nth_drone,0]) < np.pi/2 and np.abs(self.rpy[nth_drone,1]) < np.pi/2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )
    
    ################################################################################

    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot.T, drag_factors*np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )
    
    ################################################################################

    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10: # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS/(4*delta_z))**2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5*(delta_xy/beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone,:]
        quat = self.quat[nth_drone,:]
        vel = self.vel[nth_drone,:]
        rpy_rates = self.rpy_rates[nth_drone,:]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL==DroneModel.CF2X or self.DRONE_MODEL==DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          quat,
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            np.dot(rotation, rpy_rates),
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone,:] = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([
            [ 0,  r, -q, p],
            [-r,  0,  p, q],
            [ q, -p,  0, r],
            [-p, -q, -r, 0]
        ]) * .5
        theta = omega_norm * dt / 2
        print(f"Shape of quat is: {quat.shape}")
        inter_quat = np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta)
        print(f"Shape of inter_quat is {inter_quat.shape}")
        quat = np.dot(inter_quat, quat)
        return quat

    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`
    
    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
    
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        p.loadURDF("samurai.urdf",
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("cube_no_rotation.urdf",
                   [-.5, -2.5, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=self.CLIENT
                   )
    
    ################################################################################
    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position 
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int 
            intermediate waypoint for drone

        """
        direction = (
            destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
            direction / distance
        )  # Normalize the direction vector
        next_step = (
            current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step
