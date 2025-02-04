from sympy import symbols, cos, sin, tan, Matrix

# def linearize_sympy(x, u, t_yaw):
# Define symbols
x1, x2, x3, x4, x5, x6, x6, x7, x8, x9 = symbols('pos_x pos_y pos_z psi theta phi')
vel_x, vel_y, vel_z = symbols('vel_x vel_y vel_z')
psi_dot, phi_dot, theta_dot = symbols('psi_dot phi_dot theta_dot')
m, g, Jx, Jy, Jz = symbols('m g Jx Jy Jz')
fx, fy, fz = symbols('fx fy fz')
taux, tauy, tauz = symbols('taux tauy tauz')

# Constants
J_x = 0.054
J_y = 0.054
J_z = 0.104

# States (theta and theta_dot)
x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]  # positions
x4, x5, x6 = x[:, 3], x[:, 4], x[:, 5]  # velocities
x7, x8, x9 = x[:, 6], x[:, 7], x[:, 8]  # angles
x10, x11, x12 = x[:, 9], x[:, 10], x[:, 11]  # angular rates

# Control inputs
u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]  # force/torque inputs

# Translational dynamics (position derivatives)
dx1 = cos(x8) * cos(x9) * x4 + (sin(x7) * sin(x8) * cos(x9) - cos(x7) * sin(x9)) * x5 + (cos(x7) * sin(x8) * cos(x9) + sin(x7) * sin(x9)) * x6
dx2 = cos(x8) * sin(x9) * x4 + (sin(x7) * sin(x8) * sin(x9) + cos(x7) * cos(x9)) * x5 + (cos(x7) * sin(x8) * sin(x9) - sin(x7) * cos(x9)) * x6
dx3 = sin(x8) * x4 - sin(x7) * cos(x8) * x5 - cos(x7) * cos(x8) * x6

# Linear acceleration (body frame)
dx4 = x12 * x5 - x11 * x6 - g * sin(x8)
dx5 = x10 * x6 - x12 * x4 + g * cos(x8) * sin(x7)
dx6 = x11 * x4 - x10 * x5 + g * cos(x8) * cos(x7) - u1 / m

# Rotational dynamics (angles)
dx7 = x10 + sin(x7) * tan(x8) * x11 + cos(x7) * tan(x8) * x12
dx8 = cos(x7) * x11 - sin(x7) * x12
dx9 = (sin(x7) / cos(x8)) * x11 - (cos(x7) / cos(x8)) * x12

# Angular accelerations
dx10 = ((J_y - J_z) / J_x) * x11 * x12 + u2 / J_x
dx11 = ((J_z - J_x) / J_y) * x10 * x12 + u3 / J_y
dx12 = ((J_x - J_y) / J_z) * x10 * x11 + t_yaw / J_z

# Concatenate dynamics into a single vector
dynamics_vector = Matrix([
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

# Example usage (uncomment and modify as needed):
# import numpy as np
# Example inputs for testing:
# linearize_sympy(np.random.rand(1, 12), np.random.rand(1, 3), t_yaw=0.1)
