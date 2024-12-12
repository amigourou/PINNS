import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

# Gravitational constant
G = 1.0

# Masses of the three bodies
m1, m2, m3 = 1.0, 1.0, 1.0

# Initial positions (equilateral triangle configuration)
r1 = np.array([1, 0])
r2 = np.array([-0.5, np.sqrt(3) / 2])
r3 = np.array([-0.5, -np.sqrt(3) / 2])

# Initial velocities (circular motion around the center of mass)
v1 = np.array([0, 0.5])
v2 = np.array([-0.5 * np.sqrt(3) / 2, -0.25])
v3 = np.array([0.5 * np.sqrt(3) / 2, -0.25])

# Convert initial conditions to a flat array for solve_ivp
initial_conditions = np.concatenate([r1, r2, r3, v1, v2, v3])

def three_body_equations(t, y):
    # Unpack positions and velocities
    r1, r2, r3 = y[0:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    # Calculate distances
    def distance(rA, rB):
        return np.linalg.norm(rA - rB)

    r12, r13, r23 = distance(r1, r2), distance(r1, r3), distance(r2, r3)

    # Compute accelerations due to gravity
    a1 = G * m2 * (r2 - r1) / r12**3 + G * m3 * (r3 - r1) / r13**3
    a2 = G * m1 * (r1 - r2) / r12**3 + G * m3 * (r3 - r2) / r23**3
    a3 = G * m1 * (r1 - r3) / r13**3 + G * m2 * (r2 - r3) / r23**3

    # Return derivative [dr1/dt, dr2/dt, dr3/dt, dv1/dt, dv2/dt, dv3/dt]
    return np.concatenate([v1, v2, v3, a1, a2, a3])

# Time span and points
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Solve the equations of motion
solution = solve_ivp(three_body_equations, t_span, initial_conditions, t_eval=t_eval, rtol=1e-9)

# Extract positions
r1_sol = solution.y[0:2]
r2_sol = solution.y[2:4]
r3_sol = solution.y[4:6]

# print(r1_sol.shape)

# Set up the figure and axis for the animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Three-Body Problem - Lagrange (Equilateral Triangle) Solution")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()

# Plot trajectories and body markers
line1, = ax.plot([], [], color="blue", label="Body 1")
line2, = ax.plot([], [], color="green", label="Body 2")
line3, = ax.plot([], [], color="red", label="Body 3")
point1, = ax.plot([], [], "o", color="blue")
point2, = ax.plot([], [], "o", color="green")
point3, = ax.plot([], [], "o", color="red")
ax.legend()

# Initialize function for animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    point3.set_data([], [])
    return line1, line2, line3, point1, point2, point3

# Animation update function
def update(frame):
    line1.set_data(r1_sol[0, :frame], r1_sol[1, :frame])
    line2.set_data(r2_sol[0, :frame], r2_sol[1, :frame])
    line3.set_data(r3_sol[0, :frame], r3_sol[1, :frame])
    point1.set_data([r1_sol[0, frame]], [r1_sol[1, frame]])  # Wrap in lists to make sequences
    point2.set_data([r2_sol[0, frame]], [r2_sol[1, frame]])
    point3.set_data([r3_sol[0, frame]], [r3_sol[1, frame]])
    return line1, line2, line3, point1, point2, point3

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True)

# Save as GIF
ani.save("three_body_lagrange.gif", dpi=80, writer=PillowWriter(fps=30))

plt.show()
