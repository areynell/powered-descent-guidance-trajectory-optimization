# TODO:
# 0) Add drag force
# 1) Make 2D animation of trajectory and control input
# 2) Scale variables for faster/more robust convergence
# 3) Use polynomial trajectories for state and controls that match collocation method
# 4) Plot midpoints as well
# 5) Try different collocation methods
# 6) Move from 2D to 3D
# 7) Add rotation of planet
# 8) Re-order functions to make reading flow easier

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from time import time

def dynamics_scaled(state_scaled, control_scaled, params, bounds, scale_factors, offsets):
    """
    Scaled dynamics in non-dimensional units.
    state = [x, y, vx, vy, m] (scaled)
    control = [ux, uy] (scaled)
    returns scaled derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt]
    computed by unscaling the inputs, calling dynamics(), and rescaling the outputs.
    """
    T = scale_factors['t']
    S_x = ca.vertcat(scale_factors['x'],
                     scale_factors['y'],
                     scale_factors['vx'],
                     scale_factors['vy'],
                     scale_factors['m'])
    S_u = ca.vertcat(scale_factors['ux'],
                     scale_factors['uy'])

    x_offsets = ca.vertcat(offsets['x'],
                           offsets['y'],
                           offsets['vx'],
                           offsets['vy'],
                           offsets['m'])
    u_offsets = ca.vertcat(offsets['ux'],
                           offsets['uy'])

    x_scaled = state_scaled
    u_scaled = control_scaled


    return (T / S_x) * dynamics(S_x*x_scaled + x_offsets, S_u*u_scaled + u_offsets, params)

def dynamics(state, control, params):
    """
    Unscaled dynamics in physical units.
    state = [x, y, vx, vy, m]
    control = [ux, uy]
    returns derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt]
    """
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    m = state[4]

    ux = control[0]
    uy = control[1]

    G = params['G']
    M = params['M']
    Isp = params['Isp']
    g0 = params['g0']

    r = ca.sqrt(x**2 + y**2)
    U_total = ca.sqrt(ux**2 + uy**2)

    dxdt = vx
    dydt = vy
    dvxdt = -G * M * (x / r**3) + ux / m
    dvydt = -G * M * (y / r**3) + uy / m
    dmdt = -U_total / (Isp * g0)

    return ca.vertcat(dxdt, dydt, dvxdt, dvydt, dmdt)

def add_hermite_simpson_dynamic_constraints(opti, X_scaled, U_scaled, dt_scaled, params, bounds, scale_factors, offsets, N):
    """
    Add dynamic constraints using Hermite-Simpson collocation.
    """

    for k in range(N):
        xk = X_scaled[:, k]
        uk = U_scaled[:, k]
        xk_next = X_scaled[:, k+1]
        uk_next = U_scaled[:, k+1]

        # Dynamics at beginning and end of interval
        fk = dynamics_scaled(xk, uk, params, bounds, scale_factors, offsets)
        fk_next = dynamics_scaled(xk_next, uk_next, params, bounds, scale_factors, offsets)

        # Midpoint state (Hermite interpolation)
        xk_mid = 0.5 * (xk + xk_next) + (dt_scaled/8.0) * (fk - fk_next)

        # Midpoint control (linear interpolation)
        uk_mid = 0.5 * (uk + uk_next)

        # Dynamics at midpoint
        fk_mid = dynamics_scaled(xk_mid, uk_mid, params, bounds, scale_factors, offsets)

        # Simpson integration
        opti.subject_to(xk_next == xk + (dt_scaled/6.0) * (fk + 4.0*fk_mid + fk_next))

def run_optimization(initial_conditions, terminal_conditions, params, bounds, tolerances, scale_factors, offsets, N):
    """
    Set up the objective function and constraints, and solve a single trajectory optimization problem.
    """

    # State dimensions
    num_states = 5  # [x, y, vx, vy, m]
    num_controls = 2  # [ux, uy]

    # Scaled initial conditions
    x0_scaled = (initial_conditions['x0'] - offsets['x']) / scale_factors['x']
    y0_scaled = (initial_conditions['y0'] - offsets['y']) / scale_factors['y']
    vx0_scaled = (initial_conditions['vx0'] - offsets['vx']) / scale_factors['vx']
    vy0_scaled = (initial_conditions['vy0'] - offsets['vy']) / scale_factors['vy']
    m0_scaled = (initial_conditions['m0'] - offsets['m']) / scale_factors['m']

    # Scaled terminal conditions
    xf_scaled = (terminal_conditions['xf'] - offsets['x']) / scale_factors['x']
    yf_scaled = (terminal_conditions['yf'] - offsets['y']) / scale_factors['y']
    vxf_scaled = (terminal_conditions['vxf'] - offsets['vx']) / scale_factors['vx']
    vyf_scaled = (terminal_conditions['vyf'] - offsets['vy']) / scale_factors['vy']
    mf_scaled = (terminal_conditions['mf'] - offsets['m']) / scale_factors['m']

    # Scaled control bounds
    u_min_scaled = bounds['u_min'] / scale_factors['ux']
    u_max_scaled = bounds['u_max'] / scale_factors['uy']

    # Scaled time horizon bounds
    t_min_scaled = bounds['t_min'] / scale_factors['t']
    t_max_scaled = bounds['t_max'] / scale_factors['t']

    # Scaled position and velocity tolerances for terminal constraints
    x_tol_scaled = tolerances['x_tol'] / scale_factors['x']
    y_tol_scaled = tolerances['y_tol'] / scale_factors['y']
    vx_tol_scaled = tolerances['vx_tol'] / scale_factors['vx']
    vy_tol_scaled = tolerances['vy_tol'] / scale_factors['vy']

    # Optimization variables
    opti = ca.Opti()

    # Scaled state variables
    X_scaled = opti.variable(num_states, N+1)
    x_scaled = X_scaled[0, :]
    y_scaled = X_scaled[1, :]
    vx_scaled = X_scaled[2, :]
    vy_scaled = X_scaled[3, :]
    m_scaled = X_scaled[4, :]

    # Scaled control variables
    U_scaled = opti.variable(num_controls, N+1)
    ux_scaled = U_scaled[0, :]
    uy_scaled = U_scaled[1, :]

    # Scaled final time
    tf_scaled = opti.variable()
    # Time step
    # NOTE: Is there a better way we can incorporate a free final time? "tdot = 1" or "time scaling"?
    dt_scaled = tf_scaled / N

    # Slack variable for surface constraint
    slack_surface = opti.variable(1, N+1)

    # Initial constraints
    opti.subject_to(x_scaled[0] == x0_scaled)
    opti.subject_to(y_scaled[0] == y0_scaled)
    opti.subject_to(vx_scaled[0] == vx0_scaled)
    opti.subject_to(vy_scaled[0] == vy0_scaled)
    opti.subject_to(m_scaled[0] == m0_scaled)

    # Final constraints
    opti.subject_to(opti.bounded(xf_scaled - x_tol_scaled, x_scaled[-1], xf_scaled + x_tol_scaled))
    opti.subject_to(opti.bounded(yf_scaled - y_tol_scaled, y_scaled[-1], yf_scaled + y_tol_scaled))
    opti.subject_to(opti.bounded(vxf_scaled - vx_tol_scaled, vx_scaled[-1], vxf_scaled + vx_tol_scaled))
    opti.subject_to(opti.bounded(vyf_scaled - vy_tol_scaled, vy_scaled[-1], vyf_scaled + vy_tol_scaled))
    opti.subject_to(opti.bounded(t_min_scaled, tf_scaled, t_max_scaled))

    # Path constraints
    # Keep vehicle above surface (r >= R), using physical units since x and y scales might differ
    x_phys = x_scaled * scale_factors['x'] + offsets['x']
    y_phys = y_scaled * scale_factors['y'] + offsets['y']
    # IMPORTANT: Normalizing by R to keep gradients well-scaled (r >= R => r/R >= 1), and using slack variable to improve convergence
    opti.subject_to(slack_surface >= 0)
    opti.subject_to(ca.sqrt(x_phys**2 + y_phys**2) / params['R'] >= 1.0 - slack_surface)

    # Thrust magnitude constraints
    opti.subject_to(ux_scaled**2 + uy_scaled**2 >= u_min_scaled**2) # Min thrust magnitude limit
    opti.subject_to(ux_scaled**2 + uy_scaled**2 <= u_max_scaled**2) # Max thrust magnitude limit

    # Mass constraints
    opti.subject_to(m_scaled >= mf_scaled)  # Keep mass above dry-mass lower bound everywhere
    # NOTE: This constraint really seems to help produce faster convergence and avoid infeasible solutions, why?
    for k in range(N):
        # Mass cannot increase over time
        opti.subject_to(m_scaled[k+1] <= m_scaled[k])

    # Dynamic equality constraints
    add_hermite_simpson_dynamic_constraints(opti, X_scaled, U_scaled, dt_scaled, params, bounds, scale_factors, offsets, N)

    # Initial guess solution: Linear interpolation between boundary conditions for states, min thrust for controls
    for i in range(N+1):
        # Linear interpolation initial guess for states
        t_frac = i / N
        opti.set_initial(x_scaled[i], (1-t_frac)*x0_scaled + t_frac*xf_scaled)
        opti.set_initial(y_scaled[i], (1-t_frac)*y0_scaled + t_frac*yf_scaled)
        opti.set_initial(vx_scaled[i], (1-t_frac)*vx0_scaled + t_frac*vxf_scaled)
        opti.set_initial(vy_scaled[i], (1-t_frac)*vy0_scaled + t_frac*vyf_scaled)
        opti.set_initial(m_scaled[i], (1-t_frac)*m0_scaled + t_frac*mf_scaled)

        # Min thrust initial guess for controls
        opti.set_initial(ux_scaled[i], u_min_scaled)
        opti.set_initial(uy_scaled[i], u_min_scaled)

    # Initialize slack on surface constraint to zero
    opti.set_initial(slack_surface, 0.0)

    # Initial guess for time horizon
    opti.set_initial(tf_scaled, (t_min_scaled + t_max_scaled) / 2.0)

    # Objective function: Maximize final mass (by minimizing fuel use) AND minimize slack
    # NOTE: Using slack variables on position and velocity seemed to worsen convergence. Better to use tolerances on terminal constraints.
    surface_slack_penalty = 1e6
    opti.minimize(-m_scaled[-1] + surface_slack_penalty * ca.sumsqr(slack_surface))

    # Set up solver
    opti.solver('ipopt', {
        'ipopt.print_level': 5,
        'expand': True,
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-6,
        'ipopt.nlp_scaling_method': 'gradient-based',
    })

    try:
        # Solve the optimization problem
        t_start = time()
        sol = opti.solve()
        t_end = time()
        t_solve = t_end - t_start

        solution_stats = opti.stats()
        iterations = len(solution_stats['iterations']['inf_pr'])

        # inf_pr_history = solution_stats['iterations']['inf_pr']
        # inf_du_history = solution_stats['iterations']['inf_du']
        # mu_history = solution_stats['iterations']['mu']
        # d_norm_history = solution_stats['iterations']['d_norm']
        # alpha_pr_history = solution_stats['iterations']['alpha_pr']
        # alpha_du_history = solution_stats['iterations']['alpha_du']

        # # print("inf_pr_history:")
        # # print(inf_pr_history)

        # # print("inf_du_history:")
        # # print(inf_du_history)

        # # 4-row subplot figure
        # fig, axs = plt.subplots(4, 1, figsize=(9, 12), sharex=False)

        # axs[0].semilogy(inf_pr_history, label='inf_pr')
        # axs[0].semilogy(inf_du_history, label='inf_du')
        # axs[0].set_title('IPOPT infeasibilities')
        # axs[0].grid(True)
        # axs[0].legend()

        # axs[1].semilogy(mu_history, label='mu')
        # axs[1].set_title('Barrier parameter (mu)')
        # axs[1].grid(True)
        # axs[1].legend()

        # axs[2].plot(alpha_pr_history, label='alpha_pr')
        # axs[2].plot(alpha_du_history, label='alpha_du')
        # axs[2].set_title('Step sizes')
        # axs[2].grid(True)
        # axs[2].legend()

        # axs[3].semilogy(d_norm_history, label='d_norm')
        # axs[3].set_title('Search direction norm')
        # axs[3].set_xlabel('Iteration')
        # axs[3].grid(True)
        # axs[3].legend()

        # plt.tight_layout()
        # plt.show()

        print("Min scaled state magnitudes:", np.min(sol.value(X_scaled), axis=1))
        print("Max scaled state magnitudes:", np.max(sol.value(X_scaled), axis=1))
        print("Min scaled control magnitudes:", np.min(sol.value(U_scaled), axis=1))
        print("Max scaled control magnitudes:", np.max(sol.value(U_scaled), axis=1))
        print("Final scaled time:", sol.value(tf_scaled))

        # Extract and unscale solution
        tf_opt = sol.value(tf_scaled) * scale_factors['t'] + offsets['t']
        x_opt = sol.value(x_scaled) * scale_factors['x'] + offsets['x']
        y_opt = sol.value(y_scaled) * scale_factors['y'] + offsets['y']
        vx_opt = sol.value(vx_scaled) * scale_factors['vx'] + offsets['vx']
        vy_opt = sol.value(vy_scaled) * scale_factors['vy'] + offsets['vy']
        m_opt = sol.value(m_scaled) * scale_factors['m'] + offsets['m']
        ux_opt = sol.value(ux_scaled) * scale_factors['ux'] + offsets['ux']
        uy_opt = sol.value(uy_scaled) * scale_factors['uy'] + offsets['uy']

        print(f"--- Run succeeded ---")
        print(f"Converged in {iterations} iterations ({t_solve:.2f} seconds)")
        print(f"Final time: {tf_opt:.2f} s")
        print(f"Final mass: {m_opt[-1]:.2f} kg")
        return {
            "success": True,
            "iterations": iterations,
            "solve_time": t_solve,
            "tf_opt": sol.value(tf_opt),
            "x_opt": sol.value(x_opt),
            "y_opt": sol.value(y_opt),
            "vx_opt": sol.value(vx_opt),
            "vy_opt": sol.value(vy_opt),
            "m_opt": sol.value(m_opt),
            "ux_opt": sol.value(ux_opt),
            "uy_opt": sol.value(uy_opt),
        }
    except Exception as e:
        print(f"--- Run failed: {e} ---")
        return {"success": False}

def plot_single_run(successful_run, N, R):
    """
    Plot the results of a single successful run.
    """

    tf_opt = successful_run["tf_opt"]
    x_opt = successful_run["x_opt"]
    y_opt = successful_run["y_opt"]
    vx_opt = successful_run["vx_opt"]
    vy_opt = successful_run["vy_opt"]
    m_opt = successful_run["m_opt"]
    ux_opt = successful_run["ux_opt"]
    uy_opt = successful_run["uy_opt"]

    print(f"Optimal time: {tf_opt:.4f} s")
    print(f"Final mass: {m_opt[-1]:.4f} kg")

    t = np.linspace(0, tf_opt, N+1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Position components
    axes[0, 0].plot(t, x_opt, label='x', linestyle='--')
    axes[0, 0].plot(t, y_opt - R, label='y - R', linestyle='--')
    axes[0, 0].plot(t, np.sqrt(x_opt**2 + y_opt**2) - R, label='r - R', color='k')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Velocity components
    axes[0, 1].plot(t, vx_opt, label='vx', linestyle='--')
    axes[0, 1].plot(t, vy_opt, label='vy', linestyle='--')
    axes[0, 1].plot(t, np.sqrt(vx_opt**2 + vy_opt**2), label='|v|', color='k')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Thrust components
    axes[1, 0].step(t, ux_opt, label='ux', linestyle='--')
    axes[1, 0].step(t, uy_opt, label='uy', linestyle='--')
    axes[1, 0].step(t, np.sqrt(ux_opt**2 + uy_opt**2), label='|u|', color='k')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Thrust [N]')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Mass
    axes[1, 1].plot(t, m_opt, label='m', color='k')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Mass [kg]')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()

    # Trajectory plot with thrust vectors
    plt.figure(figsize=(10, 10))
    plt.plot(x_opt, y_opt, 'b-', label='Trajectory')
    plt.plot(x_opt[0], y_opt[0], 'go', label='Start')
    plt.plot(x_opt[-1], y_opt[-1], 'ro', label='End')

    # Draw thrust vectors
    thrust_scale = 0.5  # Scale factor for visualization
    for i in range(N):
        # Calculate thrust magnitude
        thrust_magnitude = np.sqrt(ux_opt[i]**2 + uy_opt[i]**2)

        # Only draw arrows if thrust magnitude is significant
        if thrust_magnitude > 1.0:  # Threshold to avoid drawing very small vectors
            plt.arrow(x_opt[i], y_opt[i],
                     thrust_scale * ux_opt[i], thrust_scale * uy_opt[i],
                     head_width=1, head_length=1, fc='r', ec='r',
                     length_includes_head=True, alpha=1.0)

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Powered Descent Trajectory with Thrust Vectors')
    plt.legend()

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
    plt.figure(figsize=(10, 10))
    for i, run in enumerate(successful_runs):
        if i == 0:
            # Add labels only for the first trajectory to avoid clutter
            plt.plot(run["x_opt"], run["y_opt"], 'b--', alpha=0.6, label='Trajectories')
            plt.plot(run["x_opt"][0], run["y_opt"][0], 'ro', label='Start Point')
            plt.plot(run["x_opt"][-1], run["y_opt"][-1], 'go', label='End Point')
        else:
            plt.plot(run["x_opt"], run["y_opt"], 'b--', alpha=0.6)
            plt.plot(run["x_opt"][0], run["y_opt"][0], 'ro')
            plt.plot(run["x_opt"][-1], run["y_opt"][-1], 'go')


    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'All {len(successful_runs)} Successful Trajectories')
    plt.legend()
    plt.show()

def main():
    # Parameters
    N = 50 # Number of control intervals
    num_runs = 100 # Number of randomized runs to perform

    # Physical parameters
    params = {
        'G': 6.672e-11, # Gravitational constant (m^3/kg/s^2)
        'M': 5.972e24, # Mass of Earth (kg)
        'R': 6.371e6, # Radius of Earth (m)
        'Isp': 300.0, # Rocket engine specific impulse (s)
        'g0': 9.81, # Standard gravity (m/s^2)
        'rho0': 1.225, # Air density at sea level (kg/m^3)
        'H': 8500.0, # Scale height of atmosphere (m)
        'Cd': 1.5, # Drag coefficient (assumed)
        'Aref': 1.0, # Reference area (m^2, assumed)
    }

    # Fixed terminal conditions for all runs
    terminal_conditions = {
        'xf': 0.0, # Final x-position (m)
        'yf': params['R'], # Final y-position (m)
        'vxf': 0.0, # Final x-velocity (m/s)
        'vyf': 0.0, # Final y-velocity (m/s)
        'mf': 10.0 # Minimum ALLOWABLE final mass (kg)
    }

    # Control and time bounds
    bounds = {
        'u_min': 1.0,      # Minimum thrust (N)
        'u_max': 5000.0,   # Maximum thrust (N)
        't_min': 1.0,      # Minimum trajectory time (s)
        't_max': 250.0     # Maximum trajectory time (s)
    }

    # Position and velocity tolerances for terminal constraints
    tolerances = {
        'x_tol': 1.0,    # Position tolerance (m)
        'y_tol': 1.0,    # Position tolerance (m)
        'vx_tol': 0.5,    # Velocity tolerance (m/s)
        'vy_tol': 0.5,    # Velocity tolerance (m/s)
    }

    results = []
    failed_initial_conditions = []
    t_start_total = time()
    np.random.seed(0)
    for i in range(num_runs):
        print(f"--- Starting run {i+1}/{num_runs} ---")
        # Randomized initial conditions
        initial_conditions = {
            'x0': np.random.uniform(-50000.0, 50000.0), # m
            'y0': params['R'] + np.random.uniform(50000.0, 100000.0), # m
            'vx0': np.random.uniform(-2500.0, 2500.0), # m/s
            'vy0': np.random.uniform(-1000.0, 0.0), # m/s
            'm0': 100.0 # kg
        }

        # initial_conditions = {
        #     'x0': -30000.0,              # x0: 30 km downrange
        #     'y0': params['R'] + 75000.0, # y0: 75 km altitude
        #     'vx0': 2000.0,               # vx0
        #     'vy0': -500.0,               # vy0
        #     'm0': 100.0                  # m0
        # }

        # Scaling factors for non-dimensionalization
        scale_factors = {
            'x': params['R'],
            'y': params['R'],
            'vx': 1000.0,     # 1000 m/s
            'vy': 1000.0,     # 1000 m/s
            'm': 100.0,       # 100 kg,
            'ux': 5000.0,     # 5 kN
            'uy': 5000.0,     # 5 kN
            't': 250.0        # 250 seconds,
        }

        # Offsets for non-dimensionalization
        offsets = {
            'x': 0.0,
            'y': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'm': 0.0,
            'ux': 0.0,
            'uy': 0.0,
            't': 0.0
        }

        print(f"Initial conditions (run {i+1}): {initial_conditions}")

        result = run_optimization(initial_conditions, terminal_conditions, params, bounds, tolerances, scale_factors, offsets, N)
        results.append(result)
        if not result['success']:
            failed_initial_conditions.append(initial_conditions)

    t_end_total = time()

    print(f"\nFinished {num_runs} runs in {t_end_total - t_start_total:.2f} seconds.")

    #  Summarize Results
    successful_runs = [r for r in results if r["success"]]
    num_success = len(successful_runs)

    print("\n--- Overall Summary ---")
    print(f"Total runs attempted: {num_runs}")
    print(f"Successful runs: {num_success} ({num_success/num_runs*100:.1f}%)")

    if failed_initial_conditions:
        print("\n--- Failed Initial Conditions ---")
        for initial_condition in failed_initial_conditions:
            print(initial_condition)

    if (num_runs == 1 and num_success == 1):
        plot_single_run(successful_runs[0], N, params['R'])
    elif (num_runs > 1 and num_success > 0):
        plot_multiple_runs(successful_runs, N, params['R'])

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=10)
    main()