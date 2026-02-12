import matplotlib.pyplot as plt
import numpy as np

def plot_single_run(trajectory, N, R):
    """
    Plot the results of a single successful run.
    """

    tf = trajectory["tf_opt"]
    x = trajectory["x_opt"]
    y = trajectory["y_opt"]
    z = trajectory["z_opt"]
    vx = trajectory["vx_opt"]
    vy = trajectory["vy_opt"]
    vz = trajectory["vz_opt"]
    m = trajectory["m_opt"]
    ux = trajectory["ux_opt"]
    uy = trajectory["uy_opt"]
    uz = trajectory["uz_opt"]

    print(f"Optimal time: {tf:.2f} s")
    print(f"Fuel consumed: {m[0] - m[-1]:.2f} kg")
    print(f"Final mass: {m[-1]:.2f} kg")

    t = np.linspace(0, tf, N+1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Position components
    axes[0, 0].plot(t, x, label='x', linestyle='--')
    axes[0, 0].plot(t, y, label='y', linestyle='--')
    axes[0, 0].plot(t, z - R, label='z - R', linestyle='--')
    axes[0, 0].plot(t, np.sqrt(x**2 + y**2 + z**2) - R, label='r - R', color='k')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    # Velocity components
    axes[0, 1].plot(t, vx, label='vx', linestyle='--')
    axes[0, 1].plot(t, vy, label='vy', linestyle='--')
    axes[0, 1].plot(t, vz, label='vz', linestyle='--')
    axes[0, 1].plot(t, np.sqrt(vx**2 + vy**2 + vz**2), label='|v|', color='k')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    # Thrust components
    axes[1, 0].step(t, ux, label='ux', linestyle='--')
    axes[1, 0].step(t, uy, label='uy', linestyle='--')
    axes[1, 0].step(t, uz, label='uz', linestyle='--')
    axes[1, 0].step(t, np.sqrt(ux**2 + uy**2 + uz**2), label='|u|', color='k')
    # TODO: Add T_min and T_max as parameters to the function and plot them here
    axes[1, 0].plot(t, [0.4*845000.0]*len(t), label='T_min', color='grey', linestyle='--')
    axes[1, 0].plot(t, [3*845000.0]*len(t), label='T_max', color='grey', linestyle='--')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Thrust (N)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    # Mass
    axes[1, 1].plot(t, m, label='m', color='k')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mass (kg)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    # Trajectory plot with thrust vectors
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'b-', label='Trajectory')
    ax.plot([x[0]], [y[0]], [z[0]], 'ro', label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'go', label='End')

    # Draw thrust vectors
    thrust_scale = 0.01  # Scale factor for visualization
    stride = max(1, N // 100) # Prevent drawing too many arrows
    for i in range(0, N, stride):
        # Calculate thrust magnitude
        thrust_magnitude = np.sqrt(ux[i]**2 + uy[i]**2 + uz[i]**2)

        # Only draw arrows if thrust magnitude is significant
        if thrust_magnitude > 1.0:  # Threshold to avoid drawing very small vectors
            ax.quiver(x[i], y[i], z[i],
                      thrust_scale * ux[i], thrust_scale * uy[i], thrust_scale * uz[i],
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
    Plot the results and statistics of multiple successful runs.
    """

    final_masses = [r["m_opt"][-1] for r in trajectories]
    final_times = [r["tf_opt"] for r in trajectories]
    iterations = [r["iterations"] for r in trajectories]

    print("\nIterations (successful runs):")
    print(f"  Min: {int(np.min(iterations))}")
    print(f"  Max: {int(np.max(iterations))}")
    print(f"  Mean: {float(np.mean(iterations)):.2f}")
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
        if i == 0:
            # Add labels only for the first trajectory to avoid clutter
            ax.plot(trajectory["x_opt"], trajectory["y_opt"], trajectory["z_opt"], 'b--', alpha=0.6, label='Trajectories')
            ax.plot([trajectory["x_opt"][0]], [trajectory["y_opt"][0]], [trajectory["z_opt"][0]], 'ro', label='Start Point')
            ax.plot([trajectory["x_opt"][-1]], [trajectory["y_opt"][-1]], [trajectory["z_opt"][-1]], 'go', label='End Point')
        else:
            ax.plot(trajectory["x_opt"], trajectory["y_opt"], trajectory["z_opt"], 'b--', alpha=0.6)
            ax.plot([trajectory["x_opt"][0]], [trajectory["y_opt"][0]], [trajectory["z_opt"][0]], 'ro')
            ax.plot([trajectory["x_opt"][-1]], [trajectory["y_opt"][-1]], [trajectory["z_opt"][-1]], 'go')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(f'All {len(trajectories)} Successful Trajectories')
    ax.legend()
    plt.show()