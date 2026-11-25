import json
import os

import numpy as np
import matplotlib.pyplot as plt

from common import State, Control

def simulate_trajectory_rk4(dynamics_fn, initial_state, control_inputs, tf, planet_params, rocket_params, N):
    """
    Simulate a trajectory using RK4 integration with a reference control input vector.

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

def compare_reference_vs_simulated_trajectories(dynamics_fn, initial_state, reference_trajectory, planet_params, rocket_params, N):
    """
    Compare a reference trajectory against an RK4-simulated trajectory using reference control inputs.

    Args:
        dynamics_fn: Function to compute state derivatives
        initial_state: Initial state array
        reference_trajectory: Reference trajectory dictionary
        planet_params: Physical parameters
        rocket_params: Rocket parameters
        N: Number of control intervals

    Returns:
        Dictionary with comparison metrics
    """

    # Extract reference trajectory
    tf = reference_trajectory["tf"]
    X = reference_trajectory["X"]
    U = reference_trajectory["U"]

    # Simulate trajectory using RK4
    u_ref = np.vstack([U[Control.X, :], U[Control.Y, :], U[Control.Z, :]])
    t_sim, X_sim = simulate_trajectory_rk4(dynamics_fn, initial_state, u_ref, tf, planet_params, rocket_params, N)

    # Compute errors
    x_ref = X[State.X, :]
    y_ref = X[State.Y, :]
    z_ref = X[State.Z, :]
    vx_ref = X[State.VX, :]
    vy_ref = X[State.VY, :]
    vz_ref = X[State.VZ, :]
    m_ref = X[State.M, :]

    x_sim = X_sim[State.X, :]
    y_sim = X_sim[State.Y, :]
    z_sim = X_sim[State.Z, :]
    vx_sim = X_sim[State.VX, :]
    vy_sim = X_sim[State.VY, :]
    vz_sim = X_sim[State.VZ, :]
    m_sim = X_sim[State.M, :]

    pos_errors = np.sqrt((x_ref - x_sim)**2 + (y_ref - y_sim)**2 + (z_ref - z_sim)**2)
    vel_errors = np.sqrt((vx_ref - vx_sim)**2 + (vy_ref - vy_sim)**2 + (vz_ref - vz_sim)**2)
    mass_errors = np.abs(m_ref - m_sim)

    # Compute metrics
    metrics = {
        'max_pos_error': np.max(pos_errors),
        'max_vel_error': np.max(vel_errors),
        'max_mass_error': np.max(mass_errors),
        'final_pos_error': pos_errors[-1],
        'final_vel_error': vel_errors[-1],
        'final_mass_error': mass_errors[-1]
    }

    print("\n--- Reference vs. Simulated Trajectory Comparison ---")
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

def plot_trajectory_comparison(reference_trajectory, simulated_comparison, N):
    """
    Plot comparison between reference and simulated trajectories.
    """
    tf_ref = reference_trajectory["tf"]
    X_ref = reference_trajectory["X"]

    t_sim = simulated_comparison['t_sim']
    X_sim = simulated_comparison['X_sim']
    pos_errors = simulated_comparison['pos_errors']
    vel_errors = simulated_comparison['vel_errors']
    mass_errors = simulated_comparison['mass_errors']

    t = np.linspace(0, tf_ref, N+1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Reference vs. Simulated Trajectory Comparison', fontsize=16)

    axes[0].plot(t, pos_errors)
    axes[0].set_ylabel('Position Error (m)')
    axes[0].grid()
    axes[0].set_title('Position Error vs. Time')

    axes[1].plot(t, vel_errors)
    axes[1].set_ylabel('Velocity Error (m/s)')
    axes[1].grid()
    axes[1].set_title('Velocity Error vs. Time')

    axes[2].plot(t, mass_errors)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Mass Error (kg)')
    axes[2].grid()
    axes[2].set_title('Mass Error vs. Time')

    # 3D trajectory comparison
    fig2 = plt.figure(figsize=(10, 10))
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(X_ref[State.X, :], X_ref[State.Y, :], X_ref[State.Z, :], 'b-', label='Reference', linewidth=2)
    ax.plot(X_sim[State.X, :], X_sim[State.Y, :], X_sim[State.Z, :], 'r--', label='Simulated (RK4)', linewidth=1.5)
    ax.plot([X_ref[State.X, 0]], [X_ref[State.Y, 0]], [X_ref[State.Z, 0]], 'ro', markersize=10, label='Start')
    ax.plot([X_ref[State.X, -1]], [X_ref[State.Y, -1]], [X_ref[State.Z, -1]], 'gs', markersize=10, label='End (Reference)')
    ax.plot([X_sim[State.X, -1]], [X_sim[State.Y, -1]], [X_sim[State.Z, -1]], 'r^', markersize=10, label='End (Sim)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Reference vs. Simulated Trajectory Comparison')
    ax.legend()

    plt.tight_layout()
    plt.show()

def save_monte_carlo_results(file_path, results, initial_conditions, settings):
    """
    Save Monte Carlo results (summary statistics plus one line per run) to a JSON file. Keeping the file under version
    control makes regressions visible with "git diff" after re-running the Monte Carlo on changed code. Solve times are
    left out on purpose, since they vary between identical runs.

    Args:
        file_path (str): Output file path (its directory is created if needed)
        results (list): Result dicts returned by the trajectory generator, one per run
        initial_conditions (list): Initial state vectors, one per run, in the same order as results
        settings (dict): Settings that produced the results (e.g. num_runs, seed, N, etc.), saved for reference
    """

    num_digits = 3

    # Per-run records
    runs = []
    num_runs = len(results)
    for i in range(num_runs):
        # A failed run may not have an iteration count (e.g. a failed Hermite-Simpson run returns just {'success': False})
        if 'iterations' in results[i]:
            iterations = int(results[i]['iterations'])
        else:
            iterations = None

        # A failed run may not have returned a trajectory at all
        if 'X' in results[i]:
            final_mass = _finite_or_none(results[i]['X'][State.M, -1], num_digits)
            tf = _finite_or_none(results[i]['tf'], num_digits)
        else:
            final_mass = None
            tf = None

        saved_initial_conditions = []
        for value in initial_conditions[i]:
            saved_initial_conditions.append(_finite_or_none(value, num_digits))

        # Each run is saved to file as a dict with the run number, success flag, iteration count, final mass, final time, and initial conditions
        run = {
            'run': i + 1,
            'success': bool(results[i]['success']),
            'iterations': iterations,
            'final_mass': final_mass,
            'tf': tf,
            'initial_conditions': saved_initial_conditions,
        }
        runs.append(run)

    # Summary statistics over the successful runs only
    successful_iterations = []
    successful_final_masses = []
    successful_tfs = []
    for i in range(num_runs):
        if not results[i]['success']:
            continue
        successful_iterations.append(results[i]['iterations'])
        successful_final_masses.append(results[i]['X'][State.M, -1])
        successful_tfs.append(results[i]['tf'])

    summary = {
        'num_runs': num_runs,
        'num_success': len(successful_tfs),
        'iterations': _statistics(successful_iterations, num_digits),
        'final_mass': _statistics(successful_final_masses, num_digits),
        'tf': _statistics(successful_tfs, num_digits),
    }

    # Settings and summary are pretty-printed.
    settings_json = _json_dumps_nested(settings)
    summary_json = _json_dumps_nested(summary)

    run_lines = []
    for run in runs:
        run_lines.append('    ' + json.dumps(run))
    runs_json = ',\n'.join(run_lines)

    lines = []
    lines.append('{')
    lines.append('  "settings": ' + settings_json + ',')
    lines.append('  "summary": ' + summary_json + ',')
    lines.append('  "runs": [')
    lines.append(runs_json)
    lines.append('  ]')
    lines.append('}')
    file_contents = '\n'.join(lines) + '\n'

    # Create the output directory if it doesn't exist yet
    directory = os.path.dirname(file_path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write(file_contents)

def _finite_or_none(value, num_digits):
    """Rounds a number for saving, mapping NaN/inf to None so the file stays valid JSON."""

    if value is None or not np.isfinite(value):
        return None
    return round(float(value), num_digits)

def _statistics(values, num_digits):
    if len(values) == 0:
        return None
    values = np.asarray(values, dtype=float)
    return {
        'mean': _finite_or_none(np.mean(values), num_digits),
        'std': _finite_or_none(np.std(values), num_digits),
        'median': _finite_or_none(np.median(values), num_digits),
        'p99': _finite_or_none(np.percentile(values, 99), num_digits),
        'min': _finite_or_none(np.min(values), num_digits),
        'max': _finite_or_none(np.max(values), num_digits),
    }

def _numpy_to_python(value):
    """Converts NumPy scalars/arrays into plain Python ints, floats and lists, which json.dumps can serialize."""

    return value.tolist()

def _json_dumps_nested(obj):
    """
    Pretty-prints obj as JSON with a 2-space indent, then indents every line after the first by a further 2 spaces,
    so the result lines up when placed as a value inside the top-level JSON object.
    """

    text = json.dumps(obj, indent=2, default=_numpy_to_python)
    text = text.replace('\n', '\n  ')
    return text
