import torch
import torch.nn as nn
from enum import Enum
import numpy as np
import control
from new_quadrotor_dynamics import QuadrotorDynamics


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

class FirstOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    first-order dynamical system in time.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          xdot = f(x, u)
        """
        super(FirstOrderDiscreteTimeSystem, self).__init__(
            continuous_time_system.nx, continuous_time_system.nu
        )
        assert callable(getattr(continuous_time_system, "forward"))
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = dt
        self.integration = integration
        self.continuous_time_system = continuous_time_system
        self.Ix = torch.eye(self.nx)

    def forward(self, x, u):
        """
        Compute x_next for a batch of x and u
        """ 
        assert x.shape[0] == u.shape[0]
        xdot = self.continuous_time_system.forward(x, u)
        if self.integration == IntegrationMethod.ExplicitEuler:
            x_next = x + xdot * self.dt
        else:
            raise NotImplementedError
        return x_next

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


class SecondOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    second-order dynamical system in time.
    """

    def __init__(
        self,
        continuous_time_system,
        dt = 0.01,
        position_integration: IntegrationMethod = IntegrationMethod.MidPoint,
        velocity_integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          qddot = f(x, u) where x = [q, qdot].
        """
        super().__init__(
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

    def output_feedback_linearized_lyapunov(self, K, L):
        """
        Given the control gain K and observer gain L, solve the discrete-time Lyapunov equation
        for the closed-loop system with the states and controls at equilibrium.
        The linearized dynamics are computed from the continuous-time system with Explicit Euler method.
        """
        x0 = self.x_equilibrium.unsqueeze(0)
        Ad, Bd = self.continuous_time_system.linearized_dynamics(
            x0, self.u_equilibrium.unsqueeze(0)
        )
        Ad = Ad.squeeze().detach().numpy()
        Bd = Bd.squeeze().detach().numpy()
        C = (
            self.continuous_time_system.linearized_observation(x0)
            .squeeze()
            .detach()
            .numpy()
        )
        Acl = np.vstack(
            (
                np.hstack((Ad + Bd @ K, -Bd @ K)),
                np.hstack((np.zeros([self.nx, self.nx]), Ad - L @ C)),
            )
        )
        Acl[np.abs(Acl) <= 1e-6] = 0
        S = control.dlyap(Acl, np.eye(2 * self.nx))
        return S

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


class QuadrotorSystem(SecondOrderDiscreteTimeSystem):

    def __init__(self, continuous_system, **kwargs):
        super().__init__(continuous_system, **kwargs) #fix dt later
        pre_mask = np.zeros(12)
        pre_mask[3:6] = 1
        self._a_mask = np.argwhere(pre_mask == 1).flatten().tolist()  # angular mask
        self._r_mask = np.argwhere(pre_mask == 1).flatten().tolist()  # rest mask
        
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
        qddot = self.continuous_time_system.forward(x, u)
        angular_vel = qddot[:, a_mask]

        print(f"angular_vel size {angular_vel.size()}")
        print(f"a_mask: {a_mask}")
        print(f"r_mask: {r_mask}")
        print(f"quat size: {quat.size()}")
        print(f"quat: {quat}")

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





