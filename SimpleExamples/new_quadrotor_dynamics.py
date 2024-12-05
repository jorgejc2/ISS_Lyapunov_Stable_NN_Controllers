import numpy as np
import torch 
from torch import Tensor
from typing import Union, Tuple, Optional
import pybullet as p
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

class QuadrotorDynamics:
    """
    Quadrotor Dynamics.
    x: state tensor
    u: control tensor, representing RPM motor values
    """

    def integrateQ(self, quat, omega):
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
        theta = omega_norm * self.time_step / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    def __init__(self, m: float = 1.4, g: float = 9.81):
        # Dimensions of state and control input
        self.nx = 12  # Number of state dimensions
        self.nu = 4   # Number of control inputs
        self.m = 0.027  # Mass of the quadrotor (kg)
        self.J_x = 2.3951e-5  # Moment of inertia around x-axis (kg·m²)
        self.J_y = 2.3951e-5  # Moment of inertia around y-axis (kg·m²)
        self.J_z = 3.2347e-5  # Moment of inertia around z-axis (kg·m²)
        self.J = torch.diag([self.J_x, self.J_y, self.J_z])
        self.J_INV = torch.linalg.inv(self.J)
        self.arm_length = 0.0397  # Distance from the center to a propeller (m)
        self.kf = 3.16e-10  # Thrust coefficient
        self.km = 7.94e-12  # Torque coefficient
        self.g = 9.81  # Gravitational acceleration (m/s²)
        # Additional properties
        self.thrust2weight = 2.25  # Thrust-to-weight ratio
        self.max_speed_kmh = 30  # Maximum speed (km/h)
        self.gnd_eff_coeff = 11.36859  # Ground effect coefficient
        self.prop_radius = 2.31348e-2  # Propeller radius (m)
        self.drag_coeff_xy = 9.1785e-7  # Drag coefficient in XY plane
        self.drag_coeff_z = 10.311e-7  # Drag coefficient in Z direction
        self.dw_coeff_1 = 2267.18  # Coefficients for downwash effect
        self.dw_coeff_2 = 0.16 
        self.dw_coeff_3 = -0.11
        self.time_step = 0.01

    def compute_rotation_matrix(quat: Tensor) -> Tensor:
        """
        Compute the rotation matrix from roll, pitch, and yaw angles.
        """
        batch = quat.shape[0]
        roll, pitch, yaw = quat[:, 0], quat[:, 1], quat[:, 2]
        c_roll, s_roll = torch.cos(roll), torch.sin(roll)
        c_pitch, s_pitch = torch.cos(pitch), torch.sin(pitch)
        c_yaw, s_yaw = torch.cos(yaw), torch.sin(yaw)

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

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Dynamics.
        x: state (batch, 12); 
        u: controller input (batch, 3).
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

        forces = u**2 * self.kf # shape (batch, nu)
        thrust = torch.zeros(batch, 3)
        thrust[:, 2] = forces.sum(1)
        
        quat = torch.concatenate((roll,pitch,yaw), dim=1) # shape (batch, 3)
        rotation = self.compute_rotation_matrix(quat) # shape (batch, 3, 3)
        # perform batch matrix multiplication
        thrust_world_frame = torch.einsum('bmm,bm->bm', rotation, thrust)
        # b = batch, m = 3
        # add singleton dimension by unsqueezing so that the subtraction is done on every batch
        force_world_frame = thrust_world_frame - torch.tensor([0, 0, self.g]).unsqueeze(0)
        acc = force_world_frame / self.m

        # Angular Dynamics
        ang_vels = torch.concatenate((ang_vel_x, ang_vel_y, ang_vel_z), dim=1) # shape (batch, 3)
        z_torque = u**2*self.km
        x_torque = (forces[:, 1] - forces[:, 3]) * self.L
        y_torque = (-forces[:,0] + forces[:, 2]) * self.L
        torques = torch.concatenate((x_torque, y_torque, z_torque), dim=1) # shape (batch, 3)
        
        torques = torques - torch.cross(ang_vels, torch.einsum('mm,bm->bm', self.J, ang_vels), dim=1)
        angular_acc = torch.einsum('bmm,bm->bm', self.J_INV, torques)
        # b = batch, m = 3

        # Update State
        vel = vel + self.time_step * angular_acc
        angular_vel = angular_vel + self.time_step * angular_acc
        pos = pos + self.time_step * vel
        quat = self._integrateQ(quat, angular_vel, self.time_step)

        # Kinematic equations
        dx = torch.zeros_like(x)
        dx[:, 0:3] = x[:, 3:6]  # Position derivatives (velocity)
        dx[:, 3:6] = x[:, 6:9]  # Angular position derivatives (angular velocity)
        dx[:, 6:9] = acc  # Velocity derivatives (acceleration)
        dx[:, 9:12] = angular_acc  # Angular velocity derivatives (angular acceleration)
        return dx

    def linearized_dynamics(self, x, u):
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


    def linearized_observation(self, x):
        batch_size = x.shape[0]
        C = torch.zeros(batch_size, self.ny, self.nx, device=x.device)
        C[:, 0] = 1
        return C

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
    
    