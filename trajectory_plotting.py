import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

from common import State, Control

def plot_single_run(trajectory, N, R):
    """
    Plot the results of a single trajectory.
    """

    tf = trajectory['tf']
    X = trajectory['X']
    U = trajectory['U']

    print(f"Optimal time: {tf:.2f} s")
    print(f"Fuel consumed: {X[State.M, 0] - X[State.M, -1]:.2f} kg")
    print(f"Final mass: {X[State.M, -1]:.2f} kg")

    t = np.linspace(0, tf, N+1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Position components
    axes[0, 0].plot(t, X[State.X, :], label='x', linestyle='--')
    axes[0, 0].plot(t, X[State.Y, :], label='y', linestyle='--')
    axes[0, 0].plot(t, X[State.Z, :] - R, label='z - R', linestyle='--')
    axes[0, 0].plot(t, np.sqrt(X[State.X, :]**2 + X[State.Y, :]**2 + X[State.Z, :]**2) - R, label='r - R', color='k')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    # Velocity components
    axes[0, 1].plot(t, X[State.VX, :], label='vx', linestyle='--')
    axes[0, 1].plot(t, X[State.VY, :], label='vy', linestyle='--')
    axes[0, 1].plot(t, X[State.VZ, :], label='vz', linestyle='--')
    axes[0, 1].plot(t, np.sqrt(X[State.VX, :]**2 + X[State.VY, :]**2 + X[State.VZ, :]**2), label='|v|', color='k')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    # Thrust components
    axes[1, 0].step(t, U[Control.X, :], label='ux', linestyle='--')
    axes[1, 0].step(t, U[Control.Y, :], label='uy', linestyle='--')
    axes[1, 0].step(t, U[Control.Z, :], label='uz', linestyle='--')
    axes[1, 0].step(t, np.sqrt(U[Control.X, :]**2 + U[Control.Y, :]**2 + U[Control.Z, :]**2), label='|u|', color='k')
    # TODO: Add T_min and T_max as parameters to the function and plot them here
    axes[1, 0].plot(t, [0.4*845000.0]*len(t), label='T_min', color='grey', linestyle='--')
    axes[1, 0].plot(t, [3*845000.0]*len(t), label='T_max', color='grey', linestyle='--')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Thrust (N)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    # Mass
    axes[1, 1].plot(t, X[State.M, :], label='m', color='k')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mass (kg)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    # Trajectory plot with thrust vectors
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[State.X, :], X[State.Y, :], X[State.Z, :] - R, 'b-', label='Trajectory')
    ax.plot([X[State.X, 0]], [X[State.Y, 0]], [X[State.Z, 0] - R], 'ro', label='Start')
    ax.plot([X[State.X, -1]], [X[State.Y, -1]], [X[State.Z, -1] - R], 'go', label='End')

    # Draw thrust vectors
    thrust_scale = 0.01  # Scale factor for visualization
    stride = max(1, N // 50) # Prevent drawing too many arrows
    for i in range(0, N, stride):
        # Calculate thrust magnitude
        thrust_magnitude = np.sqrt(U[Control.X, i]**2 + U[Control.Y, i]**2 + U[Control.Z, i]**2)

        # Only draw arrows if thrust magnitude is significant
        if thrust_magnitude > 1.0:  # Threshold to avoid drawing very small vectors
            ax.quiver(X[State.X, i], X[State.Y, i], X[State.Z, i] - R,
                      thrust_scale * U[Control.X, i], thrust_scale * U[Control.Y, i], thrust_scale * U[Control.Z, i],
                      color='r', length=thrust_magnitude * thrust_scale, normalize=True, alpha=1.0)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Powered Descent Trajectory with Thrust Vectors')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_multiple_runs(trajectories):
    """
    Plot the results and statistics of multiple trajectories.
    """

    final_masses = [traj["X"][State.M, -1] for traj in trajectories]
    final_times = [traj["tf"] for traj in trajectories]
    iterations = [traj["iterations"] for traj in trajectories]

    print("\nIterations (successful runs):")
    print(f"  Min: {int(np.min(iterations))}")
    print(f"  Max: {int(np.max(iterations))}")
    print(f"  Mean: {float(np.mean(iterations)):.2f}")
    print(f"  Median: {float(np.median(iterations)):.2f}")
    print(f"  Standard Deviation: {float(np.std(iterations)):.2f}")

    print("\nFinal Mass (kg):")
    print(f"  Avg: {np.mean(final_masses):.2f}")
    print(f"  Std: {np.std(final_masses):.2f}")
    print(f"  Min: {np.min(final_masses):.2f}")
    print(f"  Max: {np.max(final_masses):.2f}")

    print("\nOptimal Time (s):")
    print(f"  Avg: {np.mean(final_times):.2f}")
    print(f"  Std: {np.std(final_times):.2f}")
    print(f"  Min: {np.min(final_times):.2f}")
    print(f"  Max: {np.max(final_times):.2f}")

    # Plot all successful trajectories
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, trajectory in enumerate(trajectories):
        x = trajectory["X"][State.X, :]
        y = trajectory["X"][State.Y, :]
        z = trajectory["X"][State.Z, :]
        if i == 0:
            # Add labels only for the first trajectory to avoid clutter
            ax.plot(x, y, z, 'b--', alpha=0.6, label='Trajectories')
            ax.plot([x[0]], [y[0]], [z[0]], 'ro', label='Start Point')
            ax.plot([x[-1]], [y[-1]], [z[-1]], 'go', label='End Point')
        else:
            ax.plot(x, y, z, 'b--', alpha=0.6)
            ax.plot([x[0]], [y[0]], [z[0]], 'ro')
            ax.plot([x[-1]], [y[-1]], [z[-1]], 'go')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(f'All {len(trajectories)} Successful Trajectories')
    ax.legend()
    plt.show()

def animate_trajectory(trajectory, N, R, thrust_scale=0.01, interval=50, num_frames=300, mode='rocket'):
    """
    mode: 'rocket' to display a simple rocket body and nozzle flame, or
          'point' to display a point mass with a thrust vector.
    """

    tf = trajectory['tf']
    X  = trajectory['X']
    U  = trajectory['U']

    # Interpolate states and controls onto dense time vector
    t_nodes = np.linspace(0, tf, N + 1)
    t_dense = np.linspace(0, tf, num_frames)

    x_spline  = CubicSpline(t_nodes, X[State.X, :])
    y_spline  = CubicSpline(t_nodes, X[State.Y, :])
    z_spline  = CubicSpline(t_nodes, X[State.Z, :] - R)
    ux_spline = CubicSpline(t_nodes, U[Control.X, :])
    uy_spline = CubicSpline(t_nodes, U[Control.Y, :])
    uz_spline = CubicSpline(t_nodes, U[Control.Z, :])

    x_interp  = x_spline(t_dense)
    y_interp  = y_spline(t_dense)
    z_interp  = z_spline(t_dense)
    ux_interp = ux_spline(t_dense)
    uy_interp = uy_spline(t_dense)
    uz_interp = uz_spline(t_dense)

    # Axis limits
    x_min, x_max = x_interp.min(), x_interp.max()
    y_min, y_max = y_interp.min(), y_interp.max()
    z_min, z_max = z_interp.min(), z_interp.max()

    fig = plt.figure(figsize=(12, 12))
    ax  = fig.add_subplot(111, projection='3d')

    # Full trajectory
    ax.plot(x_interp, y_interp, z_interp, 'b-', alpha=0.2, label='Full trajectory')
    ax.plot([x_interp[0]], [y_interp[0]], [z_interp[0]], 'ro', label='Start')
    ax.plot([x_interp[-1]], [y_interp[-1]], [z_interp[-1]], 'go', label='End')

    # Static plane projections of full trajectory
    ax.plot(x_interp, y_interp, [z_min]*num_frames, color='b', alpha=0.08)  # x-y plane
    ax.plot(x_interp, [y_max]*num_frames, z_interp, color='b', alpha=0.08)  # x-z plane
    ax.plot([x_min]*num_frames, y_interp, z_interp, color='b', alpha=0.08)  # y-z plane

    # Creating empty plot objects for trajectory path and current position
    path = ax.plot([], [], [], 'b-', linewidth=1.5)[0]
    current_position = ax.plot([], [], [], 'ko', markersize=6)[0]

    # Creating empty plot objects for plane projections of current position
    proj_xy = ax.plot([], [], [], 'k.', markersize=10, alpha=0.3)[0]
    proj_xz = ax.plot([], [], [], 'k.', markersize=10, alpha=0.3)[0]
    proj_yz = ax.plot([], [], [], 'k.', markersize=10, alpha=0.3)[0]

    if mode == 'rocket':
        rocket_flame = ax.plot([], [], [], color='red', linewidth=3)[0]
        rocket_body = ax.plot([], [], [], color='black', linewidth=8, solid_capstyle='round')[0]
    elif mode == 'point':
        thrust_vectors = [None]
    else:
        raise ValueError("Invalid mode. Use 'rocket' or 'point'.")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Force equal aspect ratio so the rocket doesn't stretch when rotating
    x_range = max(x_max - x_min, 1.0e-6)
    y_range = max(y_max - y_min, 1.0e-6)
    z_range = max(z_max - z_min, 1.0e-6)
    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Animated Powered Descent')
    ax.legend()

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def init():
        return []

    def update(frame):
        # Trajectory path and current position
        path.set_data(x_interp[:frame+1], y_interp[:frame+1])
        path.set_3d_properties(z_interp[:frame+1])
        current_position.set_data([x_interp[frame]], [y_interp[frame]])
        current_position.set_3d_properties([z_interp[frame]])

        # Plane projections of current position
        proj_xy.set_data([x_interp[frame]], [y_interp[frame]])
        proj_xz.set_data([x_interp[frame]], [y_max])
        proj_yz.set_data([x_min], [y_interp[frame]])

        proj_xy.set_3d_properties([z_min])
        proj_xz.set_3d_properties([z_interp[frame]])
        proj_yz.set_3d_properties([z_interp[frame]])

        thrust_mag = np.sqrt(ux_interp[frame]**2 + uy_interp[frame]**2 + uz_interp[frame]**2)
        artists = [path, current_position, proj_xy, proj_xz, proj_yz, time_text]

        if mode == 'rocket':
            # Calculate orientation (default to pointing up if no thrust)
            if thrust_mag > 1.0e-6:
                dx = ux_interp[frame] / thrust_mag
                dy = uy_interp[frame] / thrust_mag
                dz = uz_interp[frame] / thrust_mag
            else:
                dx = 0.0
                dy = 0.0
                dz = 1.0

            # Scale rocket body length to be 5% of the maximum axis range for visibility
            rocket_length = max(x_range, y_range, z_range) * 0.05

            # Rocket body (pointing along thrust vector)
            rocket_body.set_data([x_interp[frame], x_interp[frame] + dx * rocket_length], [y_interp[frame], y_interp[frame] + dy * rocket_length])
            rocket_body.set_3d_properties([z_interp[frame], z_interp[frame] + dz * rocket_length])

            # Rocket flame (pointing opposite to thrust vector)
            flame_length = thrust_mag * thrust_scale if thrust_mag > 1.0 else 0.0
            rocket_flame.set_data([x_interp[frame], x_interp[frame] - dx * flame_length], [y_interp[frame], y_interp[frame] - dy * flame_length])
            rocket_flame.set_3d_properties([z_interp[frame], z_interp[frame] - dz * flame_length])

            artists.extend([rocket_body, rocket_flame])
        elif mode == 'point':
            # Thrust quiver
            if thrust_vectors[0] is not None:
                thrust_vectors[0].remove()
                thrust_vectors[0] = None

            if thrust_mag > 1.0:
                thrust_vectors[0] = ax.quiver(
                    x_interp[frame], y_interp[frame], z_interp[frame],
                    ux_interp[frame], uy_interp[frame], uz_interp[frame],
                    length=thrust_mag * thrust_scale,
                    normalize=True, color='r', alpha=1.0
                )
                artists.append(thrust_vectors[0])
        else:
            raise ValueError("Invalid mode. Use 'rocket' or 'point'.")

        time_text.set_text(f't = {t_dense[frame]:.1f} s')
        return artists

    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=interval, blit=False)

    plt.tight_layout()
    plt.show()
    return anim