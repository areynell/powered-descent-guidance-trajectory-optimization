# TODO:
# 0) Add drag force
# 1) Make 2D animation of trajectory and control input
# 2) Scale variables for faster/more robust convergence
# 3) Use Hermite interpolation polynomial to get smoother trajectory plots
# 4) Plot midpoints as well

# TODO: Use polynomial trajectories for state and controls that match collocation method
# TODO: Ensure that different scaling is applied to x and y position (e.g. x ~ O(0) whereas y ~ O(R), so can't use same scale for x and y)

from pdb import run
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from time import time

# System dynamics
def dynamics(state, control, params):
    """
    Compute the dynamics of the system.
    """

    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    m = state[4]

    ux = control[0]
    uy = control[1]

    # TODO: Scale variables correctly
    # TODO: Make position w.r.t. local point on surface of Earth, instead of w.r.t. center of Earth

    # Unpack parameters
    G, M, R, Isp, g0, rho0, H, Cd, A = params
    # eps_radius = 1e-9      # smoothing for |r| at r≈0
    # eps_vel = 1e-9      # smoothing for |v| at v≈0
    eps_thrust = 1e-6   # smoothing to avoid NaN gradients at u=0

    # Two-body gravity
    ax_gravity = -G*M*x/(x**2 + y**2)**(3/2)
    ay_gravity = -G*M*y/(x**2 + y**2)**(3/2)

    # # Atmospheric density (exponential with scale height)
    # r = ca.sqrt(x**2 + y**2 + eps_radius)
    # h = r - R
    # rho = rho0 * ca.exp(-h / H)

    # # Aerodynamic drag: Fd = 0.5*rho*Cd*A*|v|*v
    # v = ca.sqrt(vx**2 + vy**2 + eps_vel)
    # ax_drag = -(0.5 * rho * Cd * A * v * vx) / m
    # ay_drag = -(0.5 * rho * Cd * A * v * vy) / m

    T = ca.sqrt(ux**2 + uy**2 + eps_thrust)  # Thrust magnitude

    # Thrust acceleration
    ax_thrust = ux / m
    ay_thrust = uy / m

    # Equations of motion
    dxdt = vx
    dydt = vy
    dvxdt = ax_gravity + ax_thrust# + ax_drag
    dvydt = ay_gravity + ay_thrust# + ay_drag
    dmdt = -T/(Isp*g0)

    return ca.vertcat(dxdt, dydt, dvxdt, dvydt, dmdt)

def add_hermite_simpson_dynamic_constraints(opti, X, U, N, dt, params):
    """
    Add dynamic constraints using Hermite-Simpson collocation.
    """

    # Dynamic constraints using Hermite-Simpson collocation
    for k in range(N):
        # State at beginning of interval
        xk = X[:, k]
        # Control at beginning of interval
        uk = U[:, k]

        # State at end of interval
        xk_next = X[:, k+1]
        # Control at end of interval (if k < N-1)
        uk_next = U[:, min(k+1, N-1)]

        # Dynamics at beginning and end of interval
        fk = dynamics(xk, uk, params)
        fk_next = dynamics(xk_next, uk_next, params)

        # Midpoint state (Hermite interpolation)
        xk_mid = 0.5 * (xk + xk_next) + (dt/8) * (fk - fk_next)

        # Midpoint control (linear interpolation)
        uk_mid = 0.5 * (uk + uk_next)

        # Dynamics at midpoint
        fk_mid = dynamics(xk_mid, uk_mid, params)

        # Simpson integration
        opti.subject_to(xk_next == xk + (dt/6) * (fk + 4*fk_mid + fk_next))

def add_gauss_lobatto_dynamic_constraints(opti, X, U, N, dt, params):
    """
    Add dynamic constraints using higher-order Gauss-Lobatto collocation.
    """

    # Dynamic constraints using higher-order Gauss-Lobatto collocation
    for k in range(N):
        # State at beginning of interval
        xk = X[:, k]
        # Control at beginning of interval
        uk = U[:, k]

        # State at end of interval
        xk_next = X[:, k+1]
        # Control at end of interval (if k < N-1)
        uk_next = U[:, min(k+1, N-1)]

        # Dynamics at beginning and end of interval
        fk = dynamics(xk, uk, params)
        fk_next = dynamics(xk_next, uk_next, params)

        # Define collocation points (Gauss-Lobatto 5th order)
        tau = [0, 0.5 - np.sqrt(15)/10, 0.5, 0.5 + np.sqrt(15)/10, 1]

        # Evaluate state at collocation points
        xc = [None] * len(tau)
        fc = [None] * len(tau)
        uc = [None] * len(tau)

        # Start point
        xc[0] = xk
        uc[0] = uk
        fc[0] = fk

        # End point
        xc[4] = xk_next
        uc[4] = uk_next
        fc[4] = fk_next

        # Interior points using Lagrange interpolation for control
        for i in range(1, 4):
            # Control at collocation point (linear interpolation)
            uc[i] = uk + tau[i] * (uk_next - uk)

            # State at collocation point (using Hermite basis functions)
            c1 = 1 - 3*tau[i]**2 + 2*tau[i]**3
            c2 = tau[i] - 2*tau[i]**2 + tau[i]**3
            c3 = 3*tau[i]**2 - 2*tau[i]**3
            c4 = -tau[i]**2 + tau[i]**3

            xc[i] = c1*xk + c2*dt*fk + c3*xk_next + c4*dt*fk_next
            fc[i] = dynamics(xc[i], uc[i], params)

        # Gauss-Lobatto weights for 5-point quadrature
        weights = [1/20, 49/180, 32/90, 49/180, 1/20]

        # Integration constraint
        integral = 0
        for i in range(len(tau)):
            integral += weights[i] * fc[i]

        opti.subject_to(xk_next == xk + dt * integral)

def run_optimization(initial_conditions, terminal_conditions, params, N):
    """
    Set up the objective function and constraints, and solve a single trajectory optimization problem.
    """

    # Initial conditions
    x0, y0, vx0, vy0, m0 = initial_conditions

    # Terminal conditions
    xf, yf, vxf, vyf, mf = terminal_conditions

    # # Constants
    G, M, R, Isp, g0, rho0, H, Cd, A = params

    # Control bounds
    u_min = -10000.0
    u_max = 10000.0

    # State dimensions
    num_states = 5  # [x, y, vx, vy, m]
    num_controls = 2  # [ux, uy]

    # Optimization variables
    opti = ca.Opti()

    # Time horizon (free)
    # t_min = 1.0
    # t_max = 100.0
    tf = opti.variable()
    # opti.subject_to(opti.bounded(t_min, tf, t_max))
    opti.subject_to(tf > 0)

    # Time step
    # NOTE: Is there a better way we can incorporate a free final time? "tdot = 1" or "time scaling"?
    dt = tf / N

    # State variables
    X = opti.variable(num_states, N+1)
    x = X[0, :]
    y = X[1, :]
    vx = X[2, :]
    vy = X[3, :]
    m = X[4, :]

    # Control variables
    U = opti.variable(num_controls, N)
    ux = U[0, :]
    uy = U[1, :]

    # Initial constraints
    opti.subject_to(x[0] == x0)
    opti.subject_to(y[0] == y0)
    opti.subject_to(vx[0] == vx0)
    opti.subject_to(vy[0] == vy0)
    opti.subject_to(m[0] == m0)

    # Final constraints
    eps_pos = 1.0     # meters
    eps_vel = 0.5     # m/s
    opti.subject_to(opti.bounded(xf - eps_pos, x[-1], xf + eps_pos))
    opti.subject_to(opti.bounded(yf - eps_pos, y[-1], yf + eps_pos))
    opti.subject_to(opti.bounded(vxf - eps_vel, vx[-1], vxf + eps_vel))
    opti.subject_to(opti.bounded(vyf - eps_vel, vy[-1], vyf + eps_vel))

    # Mass constraints (mass cannot increase)
    # NOTE: This constraint really seems to help produce faster convergence and avoid infeasible solutions, why?
    for k in range(N):
        opti.subject_to(m[k+1] <= m[k])

    # Control constraints
    opti.subject_to(opti.bounded(u_min, ux, u_max))
    opti.subject_to(opti.bounded(u_min, uy, u_max))

    # Path constraints
    opti.subject_to(m >= mf)  # Keep mass above dry-mass lower bound everywhere
    # opti.subject_to(x**2 + y**2 >= R**2) # Keep vehicle above surface
    # opti.subject_to(ux**2 + uy**2 <= u_max**2)  # Thrust magnitude limit

    add_hermite_simpson_dynamic_constraints(opti, X, U, N, dt, params)
    # add_gauss_lobatto_dynamic_constraints(opti, X, U, N, dt, params)

    # Objective: Maximize final mass (by minimizing negative final mass)
    opti.minimize(-m[-1])

    # Initial guess for decision variables
    # NOTE: Using a more reasonable guess (e.g. 20 seconds) actually produces more infeasible results, why?
    opti.set_initial(tf, 100.0)

    for i in range(N+1):
        # Linear interpolation for state initial guess
        t_frac = i / N
        opti.set_initial(x[i], (1-t_frac)*x0 + t_frac*xf)
        opti.set_initial(y[i], (1-t_frac)*y0 + t_frac*yf)
        opti.set_initial(vx[i], (1-t_frac)*vx0 + t_frac*vxf)
        opti.set_initial(vy[i], (1-t_frac)*vy0 + t_frac*vyf)
        opti.set_initial(m[i], (1-t_frac)*m0 + t_frac*mf)

        if i < N:
            # opti.set_initial(ux[i], 1e-6)
            # opti.set_initial(uy[i], 1e-6)
            opti.set_initial(ux[i], 0.0)
            opti.set_initial(uy[i], 0.0)

    # Set up solver
    opti.solver('ipopt', {
        'ipopt.print_level': 5,
        'expand': True,
        'ipopt.max_iter': 5000,
        'ipopt.tol': 1e-6
    })

    try:
        # Solve the optimization problem
        sol = opti.solve()

        solution_stats = opti.stats()
        inf_pr_history = solution_stats['iterations']['inf_pr']
        inf_du_history = solution_stats['iterations']['inf_du']
        mu_history = solution_stats['iterations']['mu']
        d_norm_history = solution_stats['iterations']['d_norm']
        alpha_pr_history = solution_stats['iterations']['alpha_pr']
        alpha_du_history = solution_stats['iterations']['alpha_du']
        iterations = len(alpha_du_history)

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

        # Extract solution
        print(f"--- Run succeeded ---")
        return {
            "success": True,
            "iters": iterations,
            "tf_opt": sol.value(tf),
            "x_opt": sol.value(x),
            "y_opt": sol.value(y),
            "vx_opt": sol.value(vx),
            "vy_opt": sol.value(vy),
            "m_opt": sol.value(m),
            "ux_opt": sol.value(ux),
            "uy_opt": sol.value(uy),
        }
    except Exception as e:
        print(f"--- Run failed: {e} ---")
        return {"success": False}

def plot_single_run(successful_run, N, R):
    """
    Plot the results of a single successful run.
    """

    # Extract solution
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

    # Time vector for plotting
    t = np.linspace(0, tf_opt, N+1)

    # Plot results
    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(t, np.sqrt(x_opt**2 + (y_opt)**2), label='Radial Position')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Radial Position [m]')
    plt.legend()
    plt.tight_layout()

    plt.subplot(5, 1, 2)
    plt.plot(t, x_opt, label='x')
    plt.plot(t, y_opt - R, label='y - R')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(t, vx_opt, label='vx')
    plt.plot(t, vy_opt, label='vy')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(t, m_opt, label='Mass')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Mass [kg]')
    plt.legend()

    plt.subplot(5, 1, 5)
    t_control = t[:-1]  # Control points are one fewer than state points
    plt.step(t_control, ux_opt, label='ux', where='post')
    plt.step(t_control, uy_opt, label='uy', where='post')
    plt.step(t_control, np.sqrt(ux_opt**2 + uy_opt**2), label='u_total', where='post')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Control [N]')
    plt.legend()

    plt.tight_layout()

    # Trajectory plot with thrust vectors
    plt.figure(figsize=(10, 10))
    plt.plot(x_opt, y_opt, 'b-', label='Trajectory')
    plt.plot(x_opt[0], y_opt[0], 'go', label='Start')
    plt.plot(x_opt[-1], y_opt[-1], 'ro', label='End')

    # Draw thrust vectors
    thrust_scale = 0.001  # Scale factor for visualization
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
    iters = [r["iters"] for r in successful_runs]

    print("\nIterations (successful runs):")
    print(f"  Min: {int(np.min(iters))}")
    print(f"  Max: {int(np.max(iters))}")
    print(f"  Mean: {float(np.mean(iters)):.1f}")
    print(f"  Standard Deviation: {float(np.std(iters)):.1f}")

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
    N = 50  # Number of control intervals
    num_runs = 1 # Number of randomized runs to perform

    G = 6.672e-11  # Gravitational constant
    M = 5.972e24   # Mass of Earth (kg)
    R = 6.371e6    # Radius of Earth (m)
    Isp = 300.0    # Rocket engine specific impulse in seconds
    g0 = 9.81      # Standard gravity in m/s^2

    # Atmosphere and aero
    rho0 = 1.225     # kg/m^3 (sea level)
    H = 8500.0       # m (scale height)
    Cd = 1.5         # drag coefficient (assumed)
    Aref = 1.0       # m^2 reference area (assumed)

    params = [G, M, R, Isp, g0, rho0, H, Cd, Aref]

    # Fixed terminal conditions for all runs
    terminal_conditions = (
        0.0,    # xf
        R,      # yf
        0.0,    # vxf
        0.0,    # vyf
        10.0    # mf (minimum allowable final mass)
    )

    results = []
    t_start_total = time()
    for i in range(num_runs):
        print(f"--- Starting run {i+1}/{num_runs} ---")
        # Randomized initial conditions
        initial_conditions = (
            np.random.uniform(-1000.0, 1000.0), # x0
            R + np.random.uniform(100.0, 1000.0), # y0
            np.random.uniform(-250.0, 250.0), # vx0
            np.random.uniform(-100.0, 0.0), # vy0
            100.0 # m0
        )

        result = run_optimization(initial_conditions, terminal_conditions, params, N)
        results.append(result)
    t_end_total = time()


    print(f"\nFinished {num_runs} runs in {t_end_total - t_start_total:.2f} seconds.")

    #  Summarize Results
    successful_runs = [r for r in results if r["success"]]
    num_success = len(successful_runs)

    print("\n--- Overall Summary ---")
    print(f"Total runs attempted: {num_runs}")
    print(f"Successful runs: {num_success} ({num_success/num_runs*100:.1f}%)")

    if (num_runs == 1 and num_success == 1):
        plot_single_run(successful_runs[0], N, R)
    elif (num_runs > 1 and num_success > 0):
        plot_multiple_runs(successful_runs, N, R)


if __name__ == "__main__":
    main()