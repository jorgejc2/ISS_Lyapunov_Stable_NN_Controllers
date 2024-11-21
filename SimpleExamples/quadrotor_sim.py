import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from quadrotor_dynamics.

to_numpy = lambda x: x.detach().cpu().numpy()  # converts Tensor to Numpy array

# Define parameters, e.g., g = 9.81, Jx = 0.054, etc.
g = 9.81
Jx, Jy, Jz = 0.054, 0.054, 0.104
m = 1.4


# Set initial conditions
initial_state = np.zeros(12)

# Define time span for the simulation
t_span = (0, 10)  # Simulate for 10 seconds
time_steps = np.linspace(t_span[0], t_span[1], 500)

# TODO: Initialize dynamical model
quadrotor_continuous = quadrotor_dynamics(m=m, l=l, beta=beta)
dynamics = SecondOrderDiscreteTimeSystem(
    pendulum_continuous,
    dt=dt,
    position_integration=position_integration,
    velocity_integration=velocity_integration,
)

# TODO: simulation the model over all time steps
for i in range(1, n_time_steps):
    time = time_steps[i]
    prev_state = states[i-1].unsqueeze(0)
    control = u[i-1].unsqueeze(0)
    new_state = dynamics.forward(prev_state, control).squeeze()
    states[i] = new_state
    print(f"time {time:.2f} sec | state {to_numpy(new_state)} | input {to_numpy(control)}")


# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])  # Plot x, y, z positions
plt.show()
