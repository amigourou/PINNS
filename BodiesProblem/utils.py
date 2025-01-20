import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

def save_gif(solution, n_frames, name):
    print("Saving gif...")
    # Extract positions
    r1_sol = solution[0:2]
    r2_sol = solution[2:4]
    # r3_sol = solution[4:6]
    all_positions = np.concatenate((r1_sol, r2_sol), axis=1)  # Combine r1 and r2
    x_min, x_max = np.min(all_positions[0]), np.max(all_positions[0])
    y_min, y_max = np.min(all_positions[1]), np.max(all_positions[1])

    # Add a small margin for better visualization
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    x_min, x_max = x_min - margin, x_max + margin
    y_min, y_max = y_min - margin, y_max + margin

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
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
        # line3.set_data(r3_sol[0, :frame], r3_sol[1, :frame])
        point1.set_data([r1_sol[0, frame]], [r1_sol[1, frame]])  # Wrap in lists to make sequences
        point2.set_data([r2_sol[0, frame]], [r2_sol[1, frame]])
        # point3.set_data([r3_sol[0, frame]], [r3_sol[1, frame]])
        return line1, line2, line3, point1, point2, point3

    # Create the animation
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

    # Save as GIF
    ani.save(name, dpi=80, writer=PillowWriter(fps=15))

def save_plot(solution, truth, times, name, opt_scatter=None):
    print("Saving gif...")
    # Extract positions
    r1_sol = solution[0:2]
    r2_sol = solution[2:4]
    # r3_sol = solution[4:6]

    r1_truth = truth[0:2]
    r2_truth = truth[2:4]
    # r3_truth = truth[4:6]

    all_positions = np.concatenate((r1_sol, r2_sol, r1_truth, r2_truth), axis=1)  # Combine r1 and r2
    x_min, x_max = np.min(all_positions[0]), np.max(all_positions[0])
    y_min, y_max = np.min(all_positions[1]), np.max(all_positions[1])

    # Add a small margin for better visualization
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    x_min, x_max = x_min - margin, x_max + margin
    y_min, y_max = y_min - margin, y_max + margin

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.plot(r1_truth[0], r1_truth[1], label = "Body1 truth", linestyle = "--", color = "#0e0e0e")
    ax.plot(r2_truth[0],r2_truth[1], label = "Body2 truth", linestyle = "--", color = "#0e0e0e")

    # Generate segments and plot with gradient colors
    def add_gradient_line(ax, x, y, cmap):
        # Create points along the trajectory to form line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create the LineCollection with color based on position along the path
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(np.linspace(0, 1, len(x)))  # Gradient control along the path
        lc.set_linewidth(2)
        ax.add_collection(lc)

    # Plot gradients for both bodies
    ax.plot(r1_sol[0], r1_sol[1], label = "Body prediction",color = "#1f77b4")
    ax.plot(r2_sol[0], r2_sol[1], color = "#1f77b4")
    # add_gradient_line(ax, r3_sol[0], r3_sol[1], cmap_body3)

    # Optionally add labels to the end points for clarity
    ax.plot(r1_sol[0][-1], r1_sol[1][-1], 'o', color='#1f77b4', label="Body1")
    ax.plot(r2_sol[0][-1], r2_sol[1][-1], 'o', color='#1f77b4', label="Body2")
    # ax.plot(r3_sol[0][-1], r3_sol[1][-1], 'o', color='green', label="Body3")

    if opt_scatter is not None:
        ax.scatter(opt_scatter[0], opt_scatter[1], color = "black")
        ax.scatter(opt_scatter[2], opt_scatter[3], color = "black")
        # ax.scatter(opt_scatter[4], opt_scatter[5], color = "black")

    ax.set_title("Three-Body Problem - Lagrange (Equilateral Triangle) Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.legend()
    fig.savefig(name)

    plt.close(fig)