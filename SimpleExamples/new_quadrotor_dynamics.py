from typing import Union, Tuple, Optional, List, Dict
import torch
import sympy as sp
from torch import Tensor
import numpy as np

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

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
        self.nq = 6
        self.nu = 4   # Number of control inputs
        self.m = m  # Mass of the quadrotor (kg)
        self.J_x = j_x  # Moment of inertia around x-axis (kg·m²)
        self.J_y = j_y  # Moment of inertia around y-axis (kg·m²)
        self.J_z = j_z  # Moment of inertia around z-axis (kg·m²)
        self.J = torch.diag(torch.tensor([self.J_x, self.J_y, self.J_z]))  # Moment of inertia matrix
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

        # By default, the equilibrium state/control will be all 0's but that can be updated
        self._x_equilibrium = torch.zeros((self.nx,))
        self._u_equilibrium = torch.zeros((self.nu,))

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """
        A forward pass of the model dynamics. Dynamics faithfully follow those described by Daniel Mellinger in
        "Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors".
        x: state (batch, 12); 
        u: controller input (batch, 4).
        """
        assert x.ndim == u.ndim == 2, "The state and control input should be batched."
        assert x.shape[1] == self.nx, f"The state should have dimension {self.nx}."
        assert u.shape[1] == self.nu, f"The control should have dimension {self.nu}."
        # States
        batch = x.shape[0]
        pos_x, pos_y, pos_z = x[:, 0], x[:, 1], x[:, 2]  # Positions
        roll, pitch, yaw = x[:, 3], x[:, 4], x[:, 5]  # Angles (orientation)
        vel_x, vel_y, vel_z = x[:, 6], x[:, 7], x[:, 8]  # Velocities
        ang_vel_x, ang_vel_y, ang_vel_z = x[:, 9], x[:, 10], x[:, 11]  # Angular velocities

        # Control inputs (motor RPMs); Commented out since we don't need to unpack u;
        # rpm_motor_1, rpm_motor_2, rpm_motor_3, rpm_motor_4 = (
        #     u[:, 0], u[:, 1], u[:, 2], u[:, 3])

        # We convert the angular speed of each motor into its vertical force. The equation for this is:
        # Fᵢ = kf ωᵢ²
        forces = u**2 * self.kf  # shape (batch, nu)
        thrust = torch.zeros(batch, 3).to(forces)
        thrust[:, 2] = forces.sum(dim=1)
        
        angles = torch.stack((roll,pitch,yaw), dim=1) # shape (batch, 3)
        rotation = QuadrotorDynamics.compute_rotation_matrix(angles) # shape (batch, 3, 3)
        # perform batch matrix multiplication
        thrust_world_frame = torch.einsum('bmn,bn->bm', rotation, thrust)
        # b = batch, m = 3, n = 3 (even though m=n, we use different letters to eliminate ambiguity)
        # add singleton dimension by unsqueezing so that the subtraction is done on every batch
        force_world_frame = thrust_world_frame - torch.tensor([0, 0, self.g]).unsqueeze(0)
        acc = force_world_frame / self.m

        # Angular Dynamics
        ang_vels = torch.stack((ang_vel_x, ang_vel_y, ang_vel_z), dim=1) # shape(batch, 3)
        # The torque or moment about the z-axis uses different constants where we now use the formula:
        # Fᵢ = km ωᵢ²
        moments = (u**2) * self.km
        z_torque = moments[:, 0] - moments[:, 1] + moments[:, 2] - moments[:, 3]
        x_torque = (forces[:, 1] - forces[:, 3]) * self.arm_length
        y_torque = (-forces[:,0] + forces[:, 2]) * self.arm_length
        torques = torch.stack((x_torque, y_torque, z_torque), dim=1) # shape (batch, 3)
        
        torques = torques - torch.cross(ang_vels, torch.einsum('mn,bn->bm', self.J, ang_vels), dim=1)
        angular_acc = torch.einsum('mn,bn->bm', self.J_INV, torques)
        # b = batch, m = 3, n = 3

        # Kinematic equations
        qddot = torch.cat((acc, angular_acc), dim=1)
        # dx = torch.cat((x[:, self.nq:], acc, angular_acc), dim=1)
        # dx = torch.zeros_like(x)
        # dx[:, 0:3] = x[:, 3:6]  # Position derivatives (velocity)
        # dx[:, 3:6] = x[:, 6:9]  # Angular position derivatives (angular velocity)
        # dx[:, 6:9] = acc  # Velocity derivatives (acceleration)
        # dx[:, 9:12] = angular_acc  # Angular velocity derivatives (angular acceleration)
        return qddot

    def linearized_dynamics(self, x_t: Tensor, u_t: Tensor) -> Tuple[Tensor, Tensor]:
        assert x_t.numel() == self.nx, f"x_t should have {self.nx} elements but instead has {x_t.numel()}."
        assert u_t.numel() == self.nu, f"u_t should have {self.nu} elements but instead has {u_t.numel()}."

        # Define parameters, state variables and control input
        m, g, km, kf = sp.symbols('m g km kf')  # Position, velocity, control
        phi, theta, psi = sp.symbols('phi theta psi')
        L = sp.symbols('L')
        p, q, r = sp.symbols('p q r')
        F1, F2, F3, F4 = sp.symbols('F1 F2 F3 F4')
        F = sp.Matrix([F1, F2, F3, F4])
        I1, I2, I3 = sp.symbols('I1 I2 I3')
        I = sp.Matrix([[I1, 0, 0], [0, I2, 0], [0, 0, I3]])
        I_inv = I.inv()
        px, py, pz, vx, vy, vz, ax, ay, az = sp.symbols('px py pz vx vy vz ax ay az')
        u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')
        u = sp.Matrix([u1, u2, u3, u4])
        F1 = u1 ** 2 * kf
        F2 = u2 ** 2 * kf
        F3 = u3 ** 2 * kf
        F4 = u4 ** 2 * kf
        M1 = u1 ** 2 * km
        M2 = u2 ** 2 * km
        M3 = u3 ** 2 * km
        M4 = u4 ** 2 * km

        # the rotation matrix from the quadrotor's body frame to the world frame
        R = sp.Matrix([[sp.cos(psi) * sp.cos(theta) - sp.sin(phi) * sp.sin(psi) * sp.sin(theta), -sp.cos(phi) * sp.sin(psi),
                     sp.cos(psi) * sp.sin(theta) + sp.cos(theta) * sp.sin(phi) * sp.sin(psi)],
                    [sp.cos(theta) * sp.sin(psi) + sp.cos(psi) * sp.sin(phi) * sp.sin(theta), sp.cos(phi) * sp.cos(psi),
                     sp.sin(psi) * sp.sin(theta) - sp.cos(psi) * sp.cos(theta) * sp.sin(phi)],
                    [-sp.cos(phi) * sp.sin(theta), sp.sin(phi), sp.cos(phi) * sp.cos(theta)]])

        # the equations describing our positional acceleration
        acc_matrix = (1 / m) * ((sp.Matrix([0, 0, -m * g]) + R @ sp.Matrix([0, 0, F1 + F2 + F3 + F4])))
        ang_matrix = I_inv @ (sp.Matrix([[L * (F2 - F4)], [L * (F3 - F1)], [M1 - M2 + M3 - M4]]) - (
                    sp.Matrix([[0, p, q], [-p, 0, r], [-q, -r, 0]]) @ (I @ sp.Matrix([p, q, r]))))

        # Form the state and derivative vector
        x = sp.Matrix([px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r])
        xdot = sp.Matrix([vx, vy, vz, p, q, r, *acc_matrix[:, 0], *ang_matrix[:, 0]])

        # Get A and B by calculating the Jacobian w.r.t. the state and control
        A_sym = xdot.jacobian(x)
        B_sym = xdot.jacobian(u)

        # Parse the values that will be substituted into A and B
        state_t = [t for t in to_numpy(x_t.flatten())]
        control_t = [t for t in to_numpy(u_t.flatten())]
        px_t, py_t, pz_t, phi_t, theta_t, psi_t, vx_t, vy_t, vz_t, p_t, q_t, r_t = state_t
        u1_t, u2_t, u3_t, u4_t = control_t
        values: Dict[sp.symbols, float] = {
            L: self.arm_length,
            I1: self.J_x,
            I2: self.J_y,
            I3: self.J_z,
            px: px_t, py: py_t, pz: pz_t,
            vx: vx_t, vy: vy_t, vz: vz_t,
            p: p_t, q: q_t, r: r_t,
            phi: phi_t, theta: theta_t, psi: psi_t,
            g: self.g,
            km: self.km,
            kf: self.kf,
            u1: u1_t, u2: u2_t, u3: u3_t, u4: u4_t,
            F1: u1 ** 2 * kf,
            F2: u2 ** 2 * kf,
            F3: u3 ** 2 * kf,
            F4: u4 ** 2 * kf,
            m: self.m
        }

        # Substitute values into the matrices
        A_numeric = A_sym.subs(values)
        B_numeric = B_sym.subs(values)
        A_numpy = np.array(A_numeric.evalf(), dtype=np.float32)
        B_numpy = np.array(B_numeric.evalf(), dtype=np.float32)
        A_t = torch.from_numpy(A_numpy).to(x_t)
        B_t = torch.from_numpy(B_numpy).to(x_t)

        # Check that this linear system is controllable
        n = A_numpy.shape[1]  # state dimension
        m = B_numpy.shape[1]  # control dimension

        # Use Cayley-Hamilton theorem to create a (n, nxm) controllability matrix whose rank tells us
        # how many states in the system are controllable.
        ctrl_matrix = B_numpy
        for i in range(1, n):
            ctrl_matrix = np.hstack((ctrl_matrix, np.linalg.matrix_power(A_numpy, i) @ B_numpy))
        assert ctrl_matrix.shape == (
        n, n * m), f"Controllability matrix does not have the proper shape of ({(n, n * m)})"

        ctrl_rank = np.linalg.matrix_rank(ctrl_matrix)
        can_ctrl = ctrl_rank >= A_numpy.shape[1]
        assert can_ctrl, f"The system is not controllable w.r.t. equilibrium\nx: \n{to_numpy(x)}\nu: \n{to_numpy(u)}"

        return A_t, B_t

    ## properties ##
    @property
    def x_equilibrium(self):
        return self._x_equilibrium

    @property
    def u_equilibrium(self):
        return self._u_equilibrium

    ## setters ##
    @x_equilibrium.setter
    def x_equilibrium(self, value: Tensor):
        assert len(value.flatten()) == self.nx, "The new value for x_equilibrium is not the correct shape."
        self._x_equilibrium = value

    @u_equilibrium.setter
    def u_equilibrium(self, value: Tensor):
        assert len(value.flatten()) == self.nu, "The new value for u_equilibrium is not the correct shape."
        self._u_equilibrium = value

    ## static methods ##
    # @staticmethod
    # def compute_rotation_matrix(euler: Tensor) -> Tensor:
    #     """
    #     Compute the rotation matrix from roll, pitch, and yaw angles.
    #     :param quat:    Tensor containing the roll, pitch, and yaw angles for all batches
    #     :return:        Rotation matrix using these angles to transform from body to world frame
    #     """
    #     batch = euler.shape[0]
    #     roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]  # unpack angles
    #
    #     # precompute trigonometric results
    #     c_roll, s_roll = torch.cos(roll), torch.sin(roll)
    #     c_pitch, s_pitch = torch.cos(pitch), torch.sin(pitch)
    #     c_yaw, s_yaw = torch.cos(yaw), torch.sin(yaw)
    #
    #     # initialize and fill rotation matrix
    #     R = torch.zeros((batch, 3, 3), device=roll.device)
    #     R[:, 0, 0] = c_yaw * c_pitch
    #     R[:, 0, 1] = c_yaw * s_pitch * s_roll - s_yaw * c_roll
    #     R[:, 0, 2] = c_yaw * s_pitch * c_roll + s_yaw * s_roll
    #     R[:, 1, 0] = s_yaw * c_pitch
    #     R[:, 1, 1] = s_yaw * s_pitch * s_roll + c_yaw * c_roll
    #     R[:, 1, 2] = s_yaw * s_pitch * c_roll - c_yaw * s_roll
    #     R[:, 2, 0] = -s_pitch
    #     R[:, 2, 1] = c_pitch * s_roll
    #     R[:, 2, 2] = c_pitch * c_roll
    #     return R
    @staticmethod
    def compute_rotation_matrix(angles: Tensor) -> Tensor:

        phi, theta, psi = angles[:, 0], angles[:, 1], angles[:, 2] # unpack angles
        ts, tc = torch.sin, torch.cos # to save space

        # calculate rotation matrix
        ret = torch.stack([
            torch.stack([tc(psi)*tc(theta) - ts(phi)*ts(psi)*ts(theta), -tc(phi)*ts(psi), tc(psi)*ts(theta) + tc(theta)*ts(phi)*ts(psi)], dim=1),
             torch.stack([tc(theta)*ts(psi) + tc(psi)*ts(phi)*ts(theta), tc(phi)*tc(psi), ts(psi)*ts(theta) - tc(psi)*tc(theta)*ts(phi)], dim=1),
          torch.stack([-tc(phi)*ts(theta), ts(phi), tc(phi)*tc(theta)], dim=1)
        ], dim=1)  # shape (batches, 3, 3)

        return ret


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
        return torch.einsum('bmn, bn->bm', inter_quat, quat)

    @staticmethod
    def quaternion_to_rotation_matrix(quat: Tensor) -> Tensor:
        """

        :param quat:
        :return:
        """
        # unpack the quaternion values
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Calculates the individual elements of the rotation matrix from the quaternion values
        a = 1 - 2*(y**2 +z**2)
        b = 2*(x*y - w*z)
        c = 2*(x*z + w*y)
        d = 2*(x*y + w*z)
        e = 1 - 2*(x**2 + z**2)
        f = 2*(y*z - w*x)
        g = 2*(x*z - w*y)
        h = 2*(y*z + w*x)
        i = 1 - 2*(x**2 + y**2)

        # Fills in the rotation matrix using the calculations from above
        rotation_matrix = torch.stack([
            torch.stack([a, b, c], dim=1),
            torch.stack([d, e, f], dim=1),
            torch.stack([g, h, i], dim=1)
        ], dim=1)  # shape (batch, 3, 3)

        return rotation_matrix
    
    @staticmethod
    def rotation_matrix_to_euler(R: Tensor)->Tensor:
        """
        Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
        Assumes XYZ rotation order.

        Parameters:
            R (numpy.ndarray): 3x3 rotation matrix.

        Returns:
            tuple: (roll, pitch, yaw) in radians.
        """

        pitch = torch.arcsin(-R[:, 2, 0])  # θ (Pitch)
        
        # if torch.abs(R[:, 2, 0]) != 1:  # Normal case (no gimbal lock)
        roll = torch.arctan2(R[:, 2, 1], R[:, 2, 2])  # φ (Roll)
        yaw = torch.arctan2(R[:, 1, 0], R[:, 0, 0])  # ψ (Yaw)
        # else:  # Gimbal lock case
        #     yaw = 0
        #     roll = np.arctan2(-R[0, 1], R[1, 1])  # φ (Roll)
        euler = torch.stack([roll, pitch, yaw], dim=1)
        return euler

    @staticmethod
    def euler_to_quaternion(euler: Tensor) -> Tensor:
        phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]  # roll, pitch, yaw
        # print(f"phi: {phi.shape}, theta: {theta.shape}, psi: {psi.shape}")
        w = torch.cos(phi / 2)*torch.cos(theta / 2)*torch.cos(psi/2) + torch.sin(phi/2)*torch.sin(theta/2)*torch.sin(psi/2)
        x = torch.sin(phi / 2)*torch.cos(theta / 2)*torch.cos(psi/2) + torch.cos(phi/2)*torch.sin(theta/2)*torch.sin(psi/2)
        y = torch.cos(phi / 2)*torch.sin(theta / 2)*torch.cos(psi/2) + torch.sin(phi/2)*torch.cos(theta/2)*torch.sin(psi/2)
        z = torch.cos(phi / 2)*torch.cos(theta / 2)*torch.sin(psi/2) + torch.sin(phi/2)*torch.sin(theta/2)*torch.cos(psi/2)

        quaternion = torch.stack([w, x, y, z], dim=1)
        return quaternion
