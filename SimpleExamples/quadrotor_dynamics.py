import torch


class QuadrotorDynamics:
    """
    The inverted pendulum, with the upright equilibrium as the state origin.
    """

    def __init__(self, m: float = 1.4, J_x: float = 0.054,J_y: float = 0.054,J_z: float = 0.104, t_yaw = 0, g: float = 9.81):
        self.nx = 12 #num of dimensions
        self.nu = 3 #dimension of controller
    
        self.J_x = J_x
        self.J_y = J_y
        self.J_z = J_z
        self.t_yaw = t_yaw

        self.g = g  # Gravity

    def forward(self, x, u):
        """
        Dynamics. x: state (batch, 12); u: controller input (batch, 3).
        """
        # States (theta, thete_dot)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]    # positions
        x4, x5, x6 = x[:, 3], x[:, 4], x[:, 5]    # velocities
        x7, x8, x9 = x[:, 6], x[:, 7], x[:, 8]    # angles
        x10, x11, x12 = x[:, 9], x[:, 10], x[:, 11]  # angular rates
        
        # Control inputs
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]    # force/torque inputs

        # Translational dynamics
        dx1 = torch.cos(x8) * torch.cos(x9) * x4 + (torch.sin(x7) * torch.sin(x8) * torch.cos(x9) - torch.cos(x7) * torch.sin(x9)) * x5 + (torch.cos(x7) * torch.sin(x8) * torch.cos(x9) + torch.sin(x7) * torch.sin(x9)) * x6
        dx2 = torch.cos(x8) * torch.sin(x9) * x4 + (torch.sin(x7) * torch.sin(x8) * torch.sin(x9) + torch.cos(x7) * torch.cos(x9)) * x5 + (torch.cos(x7) * torch.sin(x8) * torch.sin(x9) - torch.sin(x7) * torch.cos(x9)) * x6
        dx3 = torch.sin(x8) * x4 - torch.sin(x7) * torch.cos(x8) * x5 - torch.cos(x7) * torch.cos(x8) * x6

        # Linear acceleration
        dx4 = x12 * x5 - x11 * x6 - self.g * torch.sin(x8)
        dx5 = x10 * x6 - x12 * x4 + self.g * torch.cos(x8) * torch.sin(x7)
        dx6 = x11 * x4 - x10 * x5 + self.g * torch.cos(x8) * torch.cos(x7) - u1 / self.m

        # Rotational dynamics
        dx7 = x10 + torch.sin(x7) * torch.tan(x8) * x11 + torch.cos(x7) * torch.tan(x8) * x12
        dx8 = torch.cos(x7) * x11 - torch.sin(x7) * x12
        dx9 = (torch.sin(x7) / torch.cos(x8)) * x11 - (torch.cos(x7) / torch.cos(x8)) * x12

        # Angular accelerations
        dx10 = (self.J_y - self.J_z) / self.J_x * x11 * x12 + u2 / self.J_x
        dx11 = (self.J_z - self.J_x) / self.J_y * x10 * x12 + u3 / self.J_y
        dx12 = (self.J_x - self.J_y) / self.J_z * x10 * x11 + (1/self.J_z)*self.t_yaw;

        # Concatenate dynamics
        dx = torch.stack([dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12], dim=1)
        
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
        return torch.zeros((self.nx,))

    @property
    def u_equilibrium(self):
        return torch.zeros((self.nu,))