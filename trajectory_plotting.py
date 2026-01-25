import matplotlib.pyplot as plt
import numpy as np

def plot_single_run(successful_run, N, R):
    """
    Plot the results of a single successful run.
    """

    tf_opt = successful_run["tf_opt"]
    x_opt = successful_run["x_opt"]
    y_opt = successful_run["y_opt"]
    z_opt = successful_run["z_opt"]
    vx_opt = successful_run["vx_opt"]
    vy_opt = successful_run["vy_opt"]
    vz_opt = successful_run["vz_opt"]
    m_opt = successful_run["m_opt"]
    ux_opt = successful_run["ux_opt"]
    uy_opt = successful_run["uy_opt"]
    uz_opt = successful_run["uz_opt"]

    print(f"Optimal time: {tf_opt:.4f} s")
    print(f"Final mass: {m_opt[-1]:.4f} kg")

    t = np.linspace(0, tf_opt, N+1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Position components
    axes[0, 0].plot(t, x_opt, label='x', linestyle='--')
    axes[0, 0].plot(t, y_opt, label='y', linestyle='--')
    axes[0, 0].plot(t, z_opt - R, label='z - R', linestyle='--')
    axes[0, 0].plot(t, np.sqrt(x_opt**2 + y_opt**2 + z_opt**2) - R, label='r - R', color='k')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].grid()
    axes[0, 0].legend()

    # Velocity components
    axes[0, 1].plot(t, vx_opt, label='vx', linestyle='--')
    axes[0, 1].plot(t, vy_opt, label='vy', linestyle='--')
    axes[0, 1].plot(t, vz_opt, label='vz', linestyle='--')
    axes[0, 1].plot(t, np.sqrt(vx_opt**2 + vy_opt**2 + vz_opt**2), label='|v|', color='k')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].grid()
    axes[0, 1].legend()

    # Thrust components
    axes[1, 0].step(t, ux_opt, label='ux', linestyle='--')
    axes[1, 0].step(t, uy_opt, label='uy', linestyle='--')
    axes[1, 0].step(t, uz_opt, label='uz', linestyle='--')
    axes[1, 0].step(t, np.sqrt(ux_opt**2 + uy_opt**2 + uz_opt**2), label='|u|', color='k')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Thrust [N]')
    axes[1, 0].grid()
    axes[1, 0].legend()

    # Mass
    axes[1, 1].plot(t, m_opt, label='m', color='k')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Mass [kg]')
    axes[1, 1].grid()
    axes[1, 1].legend()

    # Trajectory plot with thrust vectors
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_opt, y_opt, z_opt, 'b-', label='Trajectory')
    ax.plot([x_opt[0]], [y_opt[0]], [z_opt[0]], 'ro', label='Start')
    ax.plot([x_opt[-1]], [y_opt[-1]], [z_opt[-1]], 'go', label='End')

    # Draw thrust vectors
    thrust_scale = 0.01  # Scale factor for visualization
    for i in range(N):
        # Calculate thrust magnitude
        thrust_magnitude = np.sqrt(ux_opt[i]**2 + uy_opt[i]**2 + uz_opt[i]**2)

        # Only draw arrows if thrust magnitude is significant
        if thrust_magnitude > 1.0:  # Threshold to avoid drawing very small vectors
            ax.quiver(x_opt[i], y_opt[i], z_opt[i],
                      thrust_scale * ux_opt[i], thrust_scale * uy_opt[i], thrust_scale * uz_opt[i],
                      color='r', length=thrust_magnitude * thrust_scale, normalize=True, alpha=1.0)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Powered Descent Trajectory with Thrust Vectors')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_multiple_runs(successful_runs, N, R):
    """
    Plot the results and statistics of multiple successful runs.
    """

    final_masses = [r["m_opt"][-1] for r in successful_runs]
    final_times = [r["tf_opt"] for r in successful_runs]
    iterations = [r["iterations"] for r in successful_runs]

    print("\nIterations (successful runs):")
    print(f"  Min: {int(np.min(iterations))}")
    print(f"  Max: {int(np.max(iterations))}")
    print(f"  Mean: {float(np.mean(iterations)):.1f}")
    print(f"  Standard Deviation: {float(np.std(iterations)):.1f}")

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
    for i, run in enumerate(successful_runs):
        if i == 0:
            # Add labels only for the first trajectory to avoid clutter
            ax.plot(run["x_opt"], run["y_opt"], run["z_opt"], 'b--', alpha=0.6, label='Trajectories')
            ax.plot([run["x_opt"][0]], [run["y_opt"][0]], [run["z_opt"][0]], 'ro', label='Start Point')
            ax.plot([run["x_opt"][-1]], [run["y_opt"][-1]], [run["z_opt"][-1]], 'go', label='End Point')
        else:
            ax.plot(run["x_opt"], run["y_opt"], run["z_opt"], 'b--', alpha=0.6)
            ax.plot([run["x_opt"][0]], [run["y_opt"][0]], [run["z_opt"][0]], 'ro')
            ax.plot([run["x_opt"][-1]], [run["y_opt"][-1]], [run["z_opt"][-1]], 'go')


    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(f'All {len(successful_runs)} Successful Trajectories')
    ax.legend()
    plt.show()