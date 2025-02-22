import torch
import sympy as sp
from torch import Tensor

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

        # We convert the angular speed of each motor into its vertical force. The equation for this is:
        # Fᵢ = kf ωᵢ²
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
        # The torque or moment about the z-axis uses different constants where we now use the formula:
        # Fᵢ = km ωᵢ²
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

    @staticmethod
    def linearize_sympy(x, u, t_yaw):
        pos_x, pos_y, pos_z, psi, theta, phi = sp.symbols('pos_x pos_y pos_z psi theta phi')
        vel_x, vel_y, vel_z = sp.symbols('vel_x vel_y vel_z')
        psi_dot, phi_dot, theta_dot = sp.symbols('psi_dot phi_dot theta_dot')
        m, g, Jx, Jy, Jz = sp.symbols('m g Jx Jy Jz')
        fx, fy, fz = sp.symbols('fx fy fz')
        taux, tauy, tauz = sp.symbols('taux tauy tauz')
        J_x: float = 0.054,
        J_y: float = 0.054
        J_z: float = 0.104

        # States (theta, thete_dot)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]  # positions
        x4, x5, x6 = x[:, 3], x[:, 4], x[:, 5]  # velocities
        x7, x8, x9 = x[:, 6], x[:, 7], x[:, 8]  # angles
        x10, x11, x12 = x[:, 9], x[:, 10], x[:, 11]  # angular rates

        # Control inputs
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]  # force/torque inputs

        # Translational dynamics (position derivatives)
        dx1 = sp.cos(x8) * sp.cos(x9) * x4 + (sp.sin(x7) * sp.sin(x8) * sp.cos(x9) - sp.cos(x7) * sp.sin(x9)) * x5 + (
                    sp.cos(x7) * sp.sin(x8) * sp.cos(x9) + sp.sin(x7) * sp.sin(x9)) * x6
        dx2 = sp.cos(x8) * sp.sin(x9) * x4 + (sp.sin(x7) * sp.sin(x8) * sp.sin(x9) + sp.cos(x7) * sp.cos(x9)) * x5 + (
                    sp.cos(x7) * sp.sin(x8) * sp.sin(x9) - sp.sin(x7) * sp.cos(x9)) * x6
        dx3 = sp.sin(x8) * x4 - sp.sin(x7) * sp.cos(x8) * x5 - sp.cos(x7) * sp.cos(x8) * x6

        # Linear acceleration (body frame)
        dx4 = x12 * x5 - x11 * x6 - g * sp.sin(x8)
        dx5 = x10 * x6 - x12 * x4 + g * sp.cos(x8) * sp.sin(x7)
        dx6 = x11 * x4 - x10 * x5 + g * sp.cos(x8) * sp.cos(x7) - u1 / m

        # Rotational dynamics (angles)
        dx7 = x10 + sp.sin(x7) * sp.tan(x8) * x11 + sp.cos(x7) * sp.tan(x8) * x12
        dx8 = sp.cos(x7) * x11 - sp.sin(x7) * x12
        dx9 = (sp.sin(x7) / sp.cos(x8)) * x11 - (sp.cos(x7) / sp.cos(x8)) * x12

        # Angular accelerations
        dx10 = ((J_y - J_z) / J_x) * x11 * x12 + u2 / J_x
        dx11 = ((J_z - J_x) / J_y) * x10 * x12 + u3 / J_y
        dx12 = ((J_x - J_y) / J_z) * x10 * x11 + t_yaw / J_z

        # Concatenate dynamics into a single vector
        dynamics_vector = sp.Matrix([
            dx1,
            dx2,
            dx3,
            dx4,
            dx5,
            dx6,
            dx7,
            dx8,
            dx9,
            dx10,
            dx11,
            dx12
        ])

        # Define variables with respect to which to compute Jacobian
        variables_A = [pos_x, pos_y, pos_z, psi, theta, phi, vel_x, vel_y, vel_z, psi_dot, phi_dot, theta_dot]
        variables_B = [taux, tauy, tauz, fz]

        # Compute Jacobian matrices
        A = dynamics_vector.jacobian(variables_A)
        B = dynamics_vector.jacobian(variables_B)

        print("A =")
        print(A)
        print("\nB =")
        print(B)

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
    def compute_rotation_matrix(euler: Tensor) -> Tensor:
        """
        Compute the rotation matrix from roll, pitch, and yaw angles.
        :param quat:    Tensor containing the roll, pitch, and yaw angles for all batches
        :return:        Rotation matrix using these angles to transform from body to world frame
        """
        batch = euler.shape[0]
        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]  # unpack angles

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
    
    ## properties ##
    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
    
    