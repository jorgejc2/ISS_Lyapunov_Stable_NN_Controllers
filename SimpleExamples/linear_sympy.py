import sympy
import sympy as sp
import numpy as np
from sympy import symbols, cos, sin, tan, Matrix

# # def linearize_sympy(x, u, t_yaw):
# # Define symbols
# x0, x1, x2, x3, x4, x5, x6, x6, x7, x8, x9, x10, x11, x12 = symbols('x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')
# u1, u2, u3 = symbols('u1 u2 u3')
# t_yaw = symbols('t_yaw')


# vel_x, vel_y, vel_z = symbols('vel_x vel_y vel_z')
# pos_x, pos_y, pos_z = symbols('pos_x pos_y pos_z')
# psi, phi, theta = symbols('psi phi theta')

# psi_dot, phi_dot, theta_dot = symbols('psi_dot phi_dot theta_dot')
# m, g, Jx, Jy, Jz = symbols('m g Jx Jy Jz')
# fx, fy, fz = symbols('fx fy fz')
# taux, tauy, tauz = symbols('taux tauy tauz')

# # Constants
# J_x = 0.054
# J_y = 0.054
# J_z = 0.104

# # States (theta and theta_dot)
# # x1, x2, x3  positions
# # x4, x5, x6  velocities
# # x7, x8, x9  angles
# # x10, x11, x12  angular rates

# # Control inputs
# # u1, u2, u3 force/torque inputs

# # Translational dynamics (position derivatives)
# dx1 = cos(x8) * cos(x9) * x4 + (sin(x7) * sin(x8) * cos(x9) - cos(x7) * sin(x9)) * x5 + (cos(x7) * sin(x8) * cos(x9) + sin(x7) * sin(x9)) * x6
# dx2 = cos(x8) * sin(x9) * x4 + (sin(x7) * sin(x8) * sin(x9) + cos(x7) * cos(x9)) * x5 + (cos(x7) * sin(x8) * sin(x9) - sin(x7) * cos(x9)) * x6
# dx3 = sin(x8) * x4 - sin(x7) * cos(x8) * x5 - cos(x7) * cos(x8) * x6

# # Linear acceleration (body frame)
# dx4 = x12 * x5 - x11 * x6 - g * sin(x8)
# dx5 = x10 * x6 - x12 * x4 + g * cos(x8) * sin(x7)
# dx6 = x11 * x4 - x10 * x5 + g * cos(x8) * cos(x7) - u1 / m

# # Rotational dynamics (angles)
# dx7 = x10 + sin(x7) * tan(x8) * x11 + cos(x7) * tan(x8) * x12
# dx8 = cos(x7) * x11 - sin(x7) * x12
# dx9 = (sin(x7) / cos(x8)) * x11 - (cos(x7) / cos(x8)) * x12

# # Angular accelerations
# dx10 = ((J_y - J_z) / J_x) * x11 * x12 + u2 / J_x
# dx11 = ((J_z - J_x) / J_y) * x10 * x12 + u3 / J_y
# dx12 = ((J_x - J_y) / J_z) * x10 * x11 + t_yaw / J_z

# # dynamics into a single vector
# dynamics_vector = Matrix([
#     dx1,
#     dx2,
#     dx3,
#     dx4,
#     dx5,
#     dx6,
#     dx7,
#     dx8,
#     dx9,
#     dx10,
#     dx11,
#     dx12
# ])

# variables_A = [pos_x, pos_y, pos_z, psi, theta, phi, vel_x, vel_y, vel_z, psi_dot, phi_dot, theta_dot]
# variables_B = [taux, tauy, tauz, fz]

# # Compute Jacobian matrices
# A = dynamics_vector.jacobian(variables_A)
# B = dynamics_vector.jacobian(variables_B)

# print("A = ")
# print(A)
# print("\nB = ")
# print(B)

# # Example usage (uncomment and modify as needed):
# # import numpy as np
# # Example inputs for testing:
# # linearize_sympy(np.random.rand(1, 12), np.random.rand(1, 3), t_yaw=0.1)

# For reference:
# ϕ := phi
# ψ := psi 
# θ : = theta

# Define state variables and control input
m, g, km, kf = sympy.symbols('m g km kf')  # Position, velocity, control
phi, theta, psi = sympy.symbols('phi theta psi')
L = sympy.symbols('L')
p, q, r = symbols('p q r')
F1, F2, F3, F4 = symbols('F1 F2 F3 F4')
F = Matrix([F1, F2, F3, F4])
I1, I2, I3 = symbols('I1 I2 I3')
I = Matrix([[I1, 0, 0], [0, I2, 0], [0, 0, I3]])
I_inv = I.inv()
px, py, pz, vx, vy, vz, ax, ay, az = symbols('px py pz vx vy vz ax ay az')
u1, u2, u3, u4 = symbols('u1 u2 u3 u4')
u = Matrix([u1, u2, u3, u4])
F1 = u1**2 *kf
F2 = u2**2 *kf
F3 = u3**2 *kf
F4 = u4**2 *kf
M1 = u1**2 *km
M2 = u2**2 *km
M3 = u3**2 *km
M4 = u4**2 *km

R = sympy.Matrix([[cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta), -cos(theta)*sin(psi), cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi)],
                 [cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta), cos(phi)*cos(psi), sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi)],
                 [-cos(phi)*sin(theta), sin(phi), cos(phi)*cos(theta)]])

acc_matrix = (1/m) * (Matrix([0, 0, -m*g]) + R@Matrix([0, 0, F1 + F2 + F3 + F4]))
ang_matrix = I_inv @ (Matrix([[L*(F2-F4)], [L*(F3-F1)], [M1 - M2 + M3 - M4]]) - (Matrix([[0, p, q], [-p , 0, r], [-q, -r, 0]])@(I@Matrix([p, q, r]))))

# Compute Jacobians for linearization
# A = f.jacobian(X)  # State matrix (df/dX)
# B = f.jacobian(sympy.Matrix([u]))  # Input matrix (df/du)

C = acc_matrix.jacobian(F)
D = ang_matrix.jacobian(Matrix([p, q, r]))
E = ang_matrix.jacobian(F)

final_b_matrix = Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    C,
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    E
])
# print(D.shape)
# print(D)

final_a_matrix = Matrix([
                [0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0, *D.row(0)],
                [0,0,0,0,0,0,0,0,0, *D.row(1)],
                [0,0,0,0,0,0,0,0,0, *D.row(2)]])


print(final_a_matrix.shape)
print(final_a_matrix)

values = {
    L: 0.0397,
    I1: 2.3951e-5,
    I2: 2.3951e-5,
    I3: 3.2347e-5,
    px: 1, py: 2, pz: 3,
    vx: 0, vy: 0, vz: 0,
    ax: 0, ay: 0, az: 0,
    # x: 0, v: 0, u: 0,
    p: 0.01, q: 0.03, r: 0.05,
    phi: 0.0, theta: 0.0, psi: 0.0,
    g: -9.81,
    km: 7.94e-12,
    kf: 3.16e-10,
    u1: 0.1, u2: 0.12, u3: 0.13, u4: 0.14,
    F1: u1**2 *kf,
    F2: u2**2 *kf,
    F3: u3**2 *kf,
    F4: u4**2 *kf
}

# Substitute values into the matrix
final_a_matrix_numeric = final_a_matrix.subs(values)
final_b_matrix_numeric = final_b_matrix.subs(values)


# Convert to a NumPy array
final_a_matrix_numpy = np.array(final_a_matrix_numeric.evalf(), dtype=np.float32)
final_b_matrix_numpy = np.array(final_b_matrix_numeric.evalf(), dtype=np.float32)

print(f"final_a_matrix (shape : {final_a_matrix_numeric.shape})\n{final_a_matrix_numeric}")
print(f"final_b_matrix (shape : {final_b_matrix_numeric.shape})\n{final_b_matrix_numeric}")


def compute_lqr():
    
    A = A_batch.squeeze(0).cpu().detach().numpy()
    B = B_batch.squeeze(0).cpu().detach().numpy()
    Q = np.eye(quadrotor_tracking_continous.nx)
    R = np.eye(quadrotor_tracking_continous.nu)
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ S)
    return K, S