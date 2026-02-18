import numpy as np
import matplotlib.pyplot as plt

def simulate_trajectory_rk4(dynamics_fn, initial_state, control_inputs, tf, planet_params, rocket_params, N):
    """
    Simulate the trajectory using RK4 integration with the optimal control inputs.

    Args:
        dynamics_fn: Function to compute state derivatives
        initial_state: Array of system states at initial time
        control_inputs: Array of control inputs at each timenode
        tf: Final time
        planet_params: Physical parameters
        rocket_params: Rocket parameters
        N: Number of control intervals

    Returns:
        t_sim: Time array
        X_sim: (M, N+1) array of simulated states
    """

    dt = tf / N
    num_states = initial_state.shape[0]

    t_sim = np.linspace(0, tf, N+1)
    X_sim = np.zeros((num_states, N+1))
    X_sim[:, 0] = initial_state
    u = control_inputs

    # Runge Kutta 4th order integration
    for k in range(N):
        # Linear interpolation of control for midpoint
        u_mid = 0.5 * (u[:, k] + u[:, k+1])

        # RK4 components
        k1 = dynamics_fn(X_sim[:, k], u[:, k], planet_params, rocket_params)
        k2 = dynamics_fn(X_sim[:, k] + 0.5 * dt * k1, u_mid, planet_params, rocket_params)
        k3 = dynamics_fn(X_sim[:, k] + 0.5 * dt * k2, u_mid, planet_params, rocket_params)
        k4 = dynamics_fn(X_sim[:, k] + dt * k3, u[:, k+1], planet_params, rocket_params)

        # State update
        X_sim[:, k+1] = X_sim[:, k] + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    return t_sim, X_sim

def compare_optimized_vs_simulated_trajectories(dynamics_fn, initial_state, optimized_trajectory, planet_params, rocket_params, N):
    """
    Compare the optimized trajectory with RK4-simulated trajectory using optimal controls.

    Args:
        dynamics_fn: Function to compute state derivatives
        initial_state: Initial state array
        optimized_trajectory: Optimization result dictionary
        planet_params: Physical parameters
        rocket_params: Rocket parameters
        N: Number of control intervals

    Returns:
        Dictionary with comparison metrics
    """

    # Extract optimized trajectory
    tf_opt = optimized_trajectory["tf_opt"]
    x_opt = optimized_trajectory["x_opt"]
    y_opt = optimized_trajectory["y_opt"]
    z_opt = optimized_trajectory["z_opt"]
    vx_opt = optimized_trajectory["vx_opt"]
    vy_opt = optimized_trajectory["vy_opt"]
    vz_opt = optimized_trajectory["vz_opt"]
    m_opt = optimized_trajectory["m_opt"]
    ux_opt = optimized_trajectory["ux_opt"]
    uy_opt = optimized_trajectory["uy_opt"]
    uz_opt = optimized_trajectory["uz_opt"]

    # Extract initial state
    X0 = np.array([initial_state['x'],
                   initial_state['y'],
                   initial_state['z'],
                   initial_state['vx'],
                   initial_state['vy'],
                   initial_state['vz'],
                   initial_state['m']])

    # Simulate trajectory using RK4
    u_opt = np.vstack([ux_opt, uy_opt, uz_opt])
    t_sim, X_sim = simulate_trajectory_rk4(dynamics_fn, X0, u_opt, tf_opt, planet_params, rocket_params, N)

    # Extract simulated states
    x_sim = X_sim[0, :]
    y_sim = X_sim[1, :]
    z_sim = X_sim[2, :]
    vx_sim = X_sim[3, :]
    vy_sim = X_sim[4, :]
    vz_sim = X_sim[5, :]
    m_sim = X_sim[6, :]

    # Compute errors
    pos_errors = np.sqrt((x_opt - x_sim)**2 + (y_opt - y_sim)**2 + (z_opt - z_sim)**2)
    vel_errors = np.sqrt((vx_opt - vx_sim)**2 + (vy_opt - vy_sim)**2 + (vz_opt - vz_sim)**2)
    mass_errors = np.abs(m_opt - m_sim)

    # Compute metrics
    metrics = {
        'max_pos_error': np.max(pos_errors),
        'max_vel_error': np.max(vel_errors),
        'max_mass_error': np.max(mass_errors),
        'final_pos_error': pos_errors[-1],
        'final_vel_error': vel_errors[-1],
        'final_mass_error': mass_errors[-1]
    }

    print("\n--- Optimized vs Simulated Trajectory Comparison ---")
    print(f"Max position error:   {metrics['max_pos_error']:.2f} m")
    print(f"Max velocity error:   {metrics['max_vel_error']:.2f} m/s")
    print(f"Max mass error:       {metrics['max_mass_error']:.2f} kg")
    print(f"Final position error: {metrics['final_pos_error']:.2f} m")
    print(f"Final velocity error: {metrics['final_vel_error']:.2f} m/s")
    print(f"Final mass error:     {metrics['final_mass_error']:.2f} kg")

    return {
        't_sim': t_sim,
        'X_sim': X_sim,
        'pos_errors': pos_errors,
        'vel_errors': vel_errors,
        'mass_errors': mass_errors,
        'metrics': metrics
    }

def plot_trajectory_comparison(optimized_trajectory, simulated_comparison, N):
    """
    Plot comparison between optimized and simulated trajectories.
    """
    tf_opt = optimized_trajectory["tf_opt"]
    x_opt = optimized_trajectory["x_opt"]
    y_opt = optimized_trajectory["y_opt"]
    z_opt = optimized_trajectory["z_opt"]

    t_sim = simulated_comparison['t_sim']
    X_sim = simulated_comparison['X_sim']
    pos_errors = simulated_comparison['pos_errors']
    vel_errors = simulated_comparison['vel_errors']
    mass_errors = simulated_comparison['mass_errors']

    x_sim = X_sim[0, :]
    y_sim = X_sim[1, :]
    z_sim = X_sim[2, :]

    t = np.linspace(0, tf_opt, N+1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, pos_errors)
    axes[0].set_ylabel('Position Error [m]')
    axes[0].grid()
    axes[0].set_title('Position Error vs. Time')

    axes[1].plot(t, vel_errors)
    axes[1].set_ylabel('Velocity Error [m/s]')
    axes[1].grid()
    axes[1].set_title('Velocity Error vs. Time')

    axes[2].plot(t, mass_errors)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Mass Error [kg]')
    axes[2].grid()
    axes[2].set_title('Mass Error vs. Time')

    # 3D trajectory comparison
    fig2 = plt.figure(figsize=(10, 10))
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(x_opt, y_opt, z_opt, 'b-', label='Optimized', linewidth=2)
    ax.plot(x_sim, y_sim, z_sim, 'r--', label='Simulated (RK4)', linewidth=1.5)
    ax.plot([x_opt[0]], [y_opt[0]], [z_opt[0]], 'ro', markersize=10, label='Start')
    ax.plot([x_opt[-1]], [y_opt[-1]], [z_opt[-1]], 'gs', markersize=10, label='End (Opt)')
    ax.plot([x_sim[-1]], [y_sim[-1]], [z_sim[-1]], 'r^', markersize=10, label='End (Sim)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Optimized vs. Simulated Trajectory Comparison')
    ax.legend()

    plt.tight_layout()
    plt.show()
