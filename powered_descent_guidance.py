# TODO:
# 0) Plot planet's surface near landing site, and add linear surface constraint instead of radial constraint
# 1) Split path constraints, initial guesses, and objective function into separate functions for clarity
# 2) Add glide-slope constraint near landing site
# 3) Change vehicle's mass, thrust and surface area to be more realistic
# 4) Make 2D animation of trajectory and control input
# 5) Use polynomial trajectories for state and controls that match collocation method
# 6) Plot midpoints as well
# 7) Try different collocation methods
# 8) Add rotation of planet
# 9) Re-order functions to make reading flow easier
# 10) Add 6-DOF dynamics

import numpy as np
import casadi as ca
from time import time

import trajectory_validation as traj_val
import trajectory_plotting as traj_plot

def dynamics_scaled(state_scaled, control_scaled, planet_params, rocket_params, bounds, scale_factors, offsets):
    """
    Scaled dynamics in non-dimensional units.
    state = [x, y, z, vx, vy, vz, m] (scaled)
    control = [ux, uy, uz] (scaled)
    returns scaled derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dm/dt]
    computed by unscaling the inputs, calling dynamics(), and rescaling the outputs.
    """
    T = scale_factors['t']
    S_x = ca.vertcat(scale_factors['x'],
                     scale_factors['y'],
                     scale_factors['z'],
                     scale_factors['vx'],
                     scale_factors['vy'],
                     scale_factors['vz'],
                     scale_factors['m'])
    S_u = ca.vertcat(scale_factors['ux'],
                     scale_factors['uy'],
                     scale_factors['uz'],
                     scale_factors['gamma'])

    x_offsets = ca.vertcat(offsets['x'],
                           offsets['y'],
                           offsets['z'],
                           offsets['vx'],
                           offsets['vy'],
                           offsets['vz'],
                           offsets['m'])
    u_offsets = ca.vertcat(offsets['ux'],
                           offsets['uy'],
                           offsets['uz'],
                           offsets['gamma'])

    x_scaled = state_scaled
    u_scaled = control_scaled

    return (T / S_x) * dynamics_ca(S_x*x_scaled + x_offsets, S_u*u_scaled + u_offsets, planet_params, rocket_params)

# TODO: Generalize dynamics function to accept both CasADi and NumPy inputs, so we don't need to update both separately
def dynamics_np(state, control, planet_params, rocket_params):
    x, y, z, vx, vy, vz, m = state
    ux, uy, uz = control

    G = planet_params['G']
    M = planet_params['M']
    R = planet_params['R']
    rho0 = planet_params['rho0']
    H = planet_params['H']

    Isp = rocket_params['Isp']
    g0 = rocket_params['g0']
    Aref = rocket_params['Aref']
    Cd = rocket_params['Cd']

    u_total = np.sqrt(ux**2 + uy**2 + uz**2)

    r = np.sqrt(x**2 + y**2 + z**2)
    h = r - R
    rho = rho0 * np.exp(-h / H)
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    D = -0.5 * rho * Cd * Aref * v * np.array([vx, vy, vz])
    Dx = D[0]
    Dy = D[1]
    Dz = D[2]

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -G * M * (x / r**3) + ux / m + Dx / m
    dvydt = -G * M * (y / r**3) + uy / m + Dy / m
    dvzdt = -G * M * (z / r**3) + uz / m + Dz / m
    dmdt = -u_total / (Isp * g0)

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dmdt])

# TODO: Generalize dynamics function to accept both CasADi and NumPy inputs, so we don't need to update both separately
def dynamics_ca(state, control, planet_params, rocket_params):
    """
    Unscaled dynamics in physical units.
    state = [x, y, z, vx, vy, vz, m]
    control = [ux, uy, uz, gamma]
    returns derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dm/dt]
    """
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    m = state[6]

    ux = control[0]
    uy = control[1]
    uz = control[2]
    gamma = control[3]

    G = planet_params['G']
    M = planet_params['M']
    R = planet_params['R']
    rho0 = planet_params['rho0']
    H = planet_params['H']

    Isp = rocket_params['Isp']
    g0 = rocket_params['g0']
    Aref = rocket_params['Aref']
    Cd = rocket_params['Cd']

    r = ca.sqrt(x**2 + y**2 + z**2)
    h = r - R
    rho = rho0 * ca.exp(-h / H)
    eps = 1e-8
    v = ca.sqrt(vx**2 + vy**2 + vz**2 + eps)
    D = -0.5 * rho * Cd * Aref * v * ca.vertcat(vx, vy, vz)
    Dx = D[0]
    Dy = D[1]
    Dz = D[2]

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -G * M * (x / r**3) + ux / m + Dx / m
    dvydt = -G * M * (y / r**3) + uy / m + Dy / m
    dvzdt = -G * M * (z / r**3) + uz / m + Dz / m
    dmdt = -gamma / (Isp * g0)

    return ca.vertcat(dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dmdt)

def add_hermite_simpson_dynamic_constraints(opti, X_scaled, U_scaled, dt_scaled, planet_params, rocket_params, bounds, scale_factors, offsets, N):
    """
    Add dynamic constraints using Hermite-Simpson collocation.
    """

    for k in range(N):
        xk = X_scaled[:, k]
        uk = U_scaled[:, k]
        xk_next = X_scaled[:, k+1]
        uk_next = U_scaled[:, k+1]

        # Dynamics at beginning and end of interval
        fk = dynamics_scaled(xk, uk, planet_params, rocket_params, bounds, scale_factors, offsets)
        fk_next = dynamics_scaled(xk_next, uk_next, planet_params, rocket_params, bounds, scale_factors, offsets)

        # Midpoint state (Hermite interpolation)
        xk_mid = 0.5 * (xk + xk_next) + (dt_scaled/8.0) * (fk - fk_next)

        # Midpoint control (linear interpolation)
        uk_mid = 0.5 * (uk + uk_next)

        # Dynamics at midpoint
        fk_mid = dynamics_scaled(xk_mid, uk_mid, planet_params, rocket_params, bounds, scale_factors, offsets)

        # Simpson integration
        opti.subject_to(xk_next == xk + (dt_scaled/6.0) * (fk + 4.0*fk_mid + fk_next))

def run_optimization(initial_conditions, terminal_conditions, planet_params, rocket_params, bounds, landing_tolerances, scale_factors, offsets, N):
    """
    Set up the objective function and constraints, and solve a single trajectory optimization problem.
    """

    # State dimensions
    num_states = 7  # [x, y, z, vx, vy, vz, m]
    num_controls = 4  # [ux, uy, uz, gamma]

    # Scaled initial conditions
    x0_scaled = (initial_conditions['x0'] - offsets['x']) / scale_factors['x']
    y0_scaled = (initial_conditions['y0'] - offsets['y']) / scale_factors['y']
    z0_scaled = (initial_conditions['z0'] - offsets['z']) / scale_factors['z']
    vx0_scaled = (initial_conditions['vx0'] - offsets['vx']) / scale_factors['vx']
    vy0_scaled = (initial_conditions['vy0'] - offsets['vy']) / scale_factors['vy']
    vz0_scaled = (initial_conditions['vz0'] - offsets['vz']) / scale_factors['vz']
    m0_scaled = (initial_conditions['m0'] - offsets['m']) / scale_factors['m']

    # Scaled terminal conditions
    xf_scaled = (terminal_conditions['xf'] - offsets['x']) / scale_factors['x']
    yf_scaled = (terminal_conditions['yf'] - offsets['y']) / scale_factors['y']
    zf_scaled = (terminal_conditions['zf'] - offsets['z']) / scale_factors['z']
    vxf_scaled = (terminal_conditions['vxf'] - offsets['vx']) / scale_factors['vx']
    vyf_scaled = (terminal_conditions['vyf'] - offsets['vy']) / scale_factors['vy']
    vzf_scaled = (terminal_conditions['vzf'] - offsets['vz']) / scale_factors['vz']
    mf_scaled = (terminal_conditions['mf'] - offsets['m']) / scale_factors['m']

    # Scaled control bounds
    u_min_scaled = (bounds['u_min'] - offsets['gamma']) / scale_factors['gamma']
    u_max_scaled = (bounds['u_max'] - offsets['gamma']) / scale_factors['gamma']

    # Scaled time horizon bounds
    t_min_scaled = (bounds['t_min'] - offsets['t']) / scale_factors['t']
    t_max_scaled = (bounds['t_max'] - offsets['t']) / scale_factors['t']

    # Scaled position and velocity tolerances for terminal constraints
    x_tol_scaled = landing_tolerances['x_tol'] / scale_factors['x']
    y_tol_scaled = landing_tolerances['y_tol'] / scale_factors['y']
    z_tol_scaled = landing_tolerances['z_tol'] / scale_factors['z']
    vx_tol_scaled = landing_tolerances['vx_tol'] / scale_factors['vx']
    vy_tol_scaled = landing_tolerances['vy_tol'] / scale_factors['vy']
    vz_tol_scaled = landing_tolerances['vz_tol'] / scale_factors['vz']

    # Optimization variables
    opti = ca.Opti()

    # Scaled state variables
    X_scaled = opti.variable(num_states, N+1)
    x_scaled = X_scaled[0, :]
    y_scaled = X_scaled[1, :]
    z_scaled = X_scaled[2, :]
    vx_scaled = X_scaled[3, :]
    vy_scaled = X_scaled[4, :]
    vz_scaled = X_scaled[5, :]
    m_scaled = X_scaled[6, :]

    # Scaled control variables
    U_scaled = opti.variable(num_controls, N+1)
    ux_scaled = U_scaled[0, :]
    uy_scaled = U_scaled[1, :]
    uz_scaled = U_scaled[2, :]
    gamma_scaled = U_scaled[3, :] # Thrust magnitude slack variable for convex relaxation

    # Scaled final time
    tf_scaled = opti.variable()
    # Time step
    # NOTE: Is there a better way we can incorporate a free final time? "tdot = 1" or "time scaling"?
    dt_scaled = tf_scaled / N

    # Slack variable for surface constraint
    slack_surface = opti.variable(1, N+1)

    # Initial state constraints
    opti.subject_to(x_scaled[0] == x0_scaled)
    opti.subject_to(y_scaled[0] == y0_scaled)
    opti.subject_to(z_scaled[0] == z0_scaled)
    opti.subject_to(vx_scaled[0] == vx0_scaled)
    opti.subject_to(vy_scaled[0] == vy0_scaled)
    opti.subject_to(vz_scaled[0] == vz0_scaled)
    opti.subject_to(m_scaled[0] == m0_scaled)

    # Final state constraints
    opti.subject_to(opti.bounded(xf_scaled - x_tol_scaled, x_scaled[-1], xf_scaled + x_tol_scaled))
    opti.subject_to(opti.bounded(yf_scaled - y_tol_scaled, y_scaled[-1], yf_scaled + y_tol_scaled))
    opti.subject_to(opti.bounded(zf_scaled - z_tol_scaled, z_scaled[-1], zf_scaled + z_tol_scaled))
    opti.subject_to(opti.bounded(vxf_scaled - vx_tol_scaled, vx_scaled[-1], vxf_scaled + vx_tol_scaled))
    opti.subject_to(opti.bounded(vyf_scaled - vy_tol_scaled, vy_scaled[-1], vyf_scaled + vy_tol_scaled))
    opti.subject_to(opti.bounded(vzf_scaled - vz_tol_scaled, vz_scaled[-1], vzf_scaled + vz_tol_scaled))

    # Time constraints
    opti.subject_to(opti.bounded(t_min_scaled, tf_scaled, t_max_scaled))

    # Path constraints
    # Keep vehicle above surface (r >= R), using physical units since x and y scales might differ
    x_phys = x_scaled * scale_factors['x'] + offsets['x']
    y_phys = y_scaled * scale_factors['y'] + offsets['y']
    z_phys = z_scaled * scale_factors['z'] + offsets['z']
    # IMPORTANT: Normalizing by R to keep gradients well-scaled (r >= R => r/R >= 1), and using slack variable to improve convergence
    opti.subject_to(slack_surface >= 0)
    opti.subject_to(ca.sqrt(x_phys**2 + y_phys**2 + z_phys**2) / planet_params['R'] >= 1.0 - slack_surface)

    # Control constraints
    opti.subject_to(opti.bounded(-u_max_scaled, ux_scaled, u_max_scaled))
    opti.subject_to(opti.bounded(-u_max_scaled, uy_scaled, u_max_scaled))
    opti.subject_to(opti.bounded(-u_max_scaled, uz_scaled, u_max_scaled))
    # Convex relaxation of thrust magnitude
    opti.subject_to(ux_scaled**2 + uy_scaled**2 + uz_scaled**2 <= gamma_scaled**2)
    opti.subject_to(opti.bounded(u_min_scaled, gamma_scaled, u_max_scaled))

    # Mass constraints
    opti.subject_to(m_scaled >= mf_scaled)  # Keep mass above dry-mass lower bound everywhere
    # NOTE: This constraint really seems to help produce faster convergence and avoid infeasible solutions, why?
    for k in range(N):
        # Mass cannot increase over time
        opti.subject_to(m_scaled[k+1] <= m_scaled[k])

    # Dynamic equality constraints
    add_hermite_simpson_dynamic_constraints(opti, X_scaled, U_scaled, dt_scaled, planet_params, rocket_params, bounds, scale_factors, offsets, N)

    # Initial guess solution: Linear interpolation between boundary conditions for states, min thrust for controls
    for i in range(N+1):
        # Linear interpolation initial guess for states
        t_frac = i / N
        opti.set_initial(x_scaled[i], (1-t_frac)*x0_scaled + t_frac*xf_scaled)
        opti.set_initial(y_scaled[i], (1-t_frac)*y0_scaled + t_frac*yf_scaled)
        opti.set_initial(z_scaled[i], (1-t_frac)*z0_scaled + t_frac*zf_scaled)
        opti.set_initial(vx_scaled[i], (1-t_frac)*vx0_scaled + t_frac*vxf_scaled)
        opti.set_initial(vy_scaled[i], (1-t_frac)*vy0_scaled + t_frac*vyf_scaled)
        opti.set_initial(vz_scaled[i], (1-t_frac)*vz0_scaled + t_frac*vzf_scaled)
        opti.set_initial(m_scaled[i], (1-t_frac)*m0_scaled + t_frac*mf_scaled)

        # Min thrust initial guess for controls
        opti.set_initial(ux_scaled[i], u_min_scaled)
        opti.set_initial(uy_scaled[i], u_min_scaled)
        opti.set_initial(uz_scaled[i], u_min_scaled)
        opti.set_initial(gamma_scaled[i], u_min_scaled)

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
        'ipopt.linear_solver': 'mumps',
        'ipopt.hessian_approximation': 'exact',
        'ipopt.nlp_scaling_method': 'gradient-based',
        # 'ipopt.nlp_scaling_method': 'none',
        # 'ipopt.mu_strategy': 'adaptive',
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
        z_opt = sol.value(z_scaled) * scale_factors['z'] + offsets['z']
        vx_opt = sol.value(vx_scaled) * scale_factors['vx'] + offsets['vx']
        vy_opt = sol.value(vy_scaled) * scale_factors['vy'] + offsets['vy']
        vz_opt = sol.value(vz_scaled) * scale_factors['vz'] + offsets['vz']
        m_opt = sol.value(m_scaled) * scale_factors['m'] + offsets['m']
        ux_opt = sol.value(ux_scaled) * scale_factors['ux'] + offsets['ux']
        uy_opt = sol.value(uy_scaled) * scale_factors['uy'] + offsets['uy']
        uz_opt = sol.value(uz_scaled) * scale_factors['uz'] + offsets['uz']
        gamma_opt = sol.value(gamma_scaled) * scale_factors['gamma'] + offsets['gamma']

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
            "z_opt": sol.value(z_opt),
            "vx_opt": sol.value(vx_opt),
            "vy_opt": sol.value(vy_opt),
            "vz_opt": sol.value(vz_opt),
            "m_opt": sol.value(m_opt),
            "ux_opt": sol.value(ux_opt),
            "uy_opt": sol.value(uy_opt),
            "uz_opt": sol.value(uz_opt),
            "gamma_opt": sol.value(gamma_opt)
        }
    except Exception as e:
        print(f"--- Run failed: {e} ---")
        return {"success": False}

def main():
    # Planet parameters
    planet_params = {
        'G': 6.672e-11, # Gravitational constant (m^3/kg/s^2)
        'M': 5.972e24, # Mass of Earth (kg)
        'R': 6.371e6, # Radius of Earth (m)
        'rho0': 1.225, # Air density at sea level (kg/m^3)
        'H': 8500.0, # Scale height of Earth's atmosphere (m)
    }

    # Rocket parameters
    rocket_params = {
        'thrust_min': 0.4*845000.0, # Minimum thrust from 1 engine (N) - ~40% throttle
        'thrust_max': 3*845000.0, # Maximum thrust from 3 engines (N) - Merlin 1D SL
        'Isp': 282.0, # Specific impulse (s)
        # 'Isp': 500.0, # Specific impulse (s)
        'g0': 9.81, # Standard gravity (m/s^2)
        'mass_start': 75000.0, # Starting mass (kg) - Falcon 9 Mass at start of boostback burn
        'mass_dry': 22200.0, # Dry mass (kg) - Falcon 9 Dry Mass
        'Aref': np.pi*(3.7/2)**2, # Reference area (m^2)
        'Cd': 1.8 # Drag coefficient (assumed)
    }

    # Fixed terminal conditions for all runs
    terminal_conditions = {
        'xf': 0.0, # Final x-position (m)
        'yf': 0.0, # Final y-position (m)
        'zf': planet_params['R'], # Final z-position (m)
        'vxf': 0.0, # Final x-velocity (m/s)
        'vyf': 0.0, # Final y-velocity (m/s)
        'vzf': 0.0, # Final z-velocity (m/s)
        'mf': rocket_params['mass_dry'] # Minimum ALLOWABLE final mass (kg)
    }

    # Control and time bounds
    bounds = {
        'u_min': rocket_params['thrust_min'], # Minimum thrust (N)
        'u_max': rocket_params['thrust_max'], # Maximum thrust (N)
        't_min': 1.0,      # Minimum trajectory time (s)
        't_max': 500.0     # Maximum trajectory time (s)
    }

    # Position and velocity tolerances for terminal constraints
    landing_tolerances = {
        'x_tol': 1.0,    # Position tolerance (m)
        'y_tol': 1.0,    # Position tolerance (m)
        'z_tol': 1.0,    # Position tolerance (m)
        'vx_tol': 0.5,    # Velocity tolerance (m/s)
        'vy_tol': 0.5,    # Velocity tolerance (m/s)
        'vz_tol': 0.5,    # Velocity tolerance (m/s)
    }

    N = 50 # Number of control intervals
    num_runs = 1 # Number of randomized runs to perform

    results = []
    failed_initial_conditions = []
    t_start_total = time()
    np.random.seed(8)
    for i in range(num_runs):
        print(f"--- Starting run {i+1}/{num_runs} ---")
        # Randomized initial conditions
        initial_conditions = {
            'x0': np.random.uniform(-50000.0, 50000.0), # m
            'y0': np.random.uniform(-50000.0, 50000.0), # m
            'z0': planet_params['R'] + np.random.uniform(50000.0, 150000.0), # m
            'vx0': np.random.uniform(-1000.0, 1000.0), # m/s
            'vy0': np.random.uniform(-1000.0, 1000.0), # m/s
            'vz0': np.random.uniform(-500.0, 500.0), # m/s
            'm0': rocket_params['mass_start'] # kg - Landing mass (Dry mass + reserve)
        }

        # --- Failed Initial Conditions ---
        # initial_conditions = {'x0': 10333.00064753613, 'y0': -14102.69257401282, 'z0': 6451802.3336159075, 'vx0': 646.6701807114714, 'vy0': 121.62380764174964, 'vz0': -152.95896172646286, 'm0': 75000.0}
        # initial_conditions = {'x0': -10988.148096549397, 'y0': 39553.273620720895, 'z0': 6430847.15719154, 'vx0': 821.1205393090227, 'vy0': 457.70904823551905, 'vz0': -372.8149990047769, 'm0': 75000.0}
        # initial_conditions = {'x0': -2229.5538019762243, 'y0': 25230.130182274952, 'z0': 6450341.862722686, 'vx0': 15.920842801517665, 'vy0': 454.84039373955056, 'vz0': -156.4300590392359, 'm0': 75000.0}
        # initial_conditions = {'x0': -20476.74314880714, 'y0': -13615.78880679213, 'z0': 6444010.141756712, 'vx0': 392.54362697458873, 'vy0': 284.2714639006099, 'vz0': -267.7434706956784, 'm0': 75000.0}
        # initial_conditions = {'x0': 28840.55161689737, 'y0': -20875.4851646929, 'z0': 6434001.974537622, 'vx0': 559.796802578966, 'vy0': 368.6994615807289, 'vz0': -204.34760760308177, 'm0': 75000.0}
        # initial_conditions = {'x0': -31518.45806321033, 'y0': 47770.549924390696, 'z0': 6433824.722904884, 'vx0': 209.2517851410796, 'vy0': 515.7533383325085, 'vz0': -79.33290732940077, 'm0': 75000.0}
        # initial_conditions = {'x0': 31116.18834065397, 'y0': 28482.906354009217, 'z0': 6449458.003926393, 'vx0': 634.4300909114133, 'vy0': 76.80522338545347, 'vz0': -192.1010130966052, 'm0': 75000.0}
        # initial_conditions = {'x0': -1481.8618443154555, 'y0': 28626.116533809356, 'z0': 6437017.691213863, 'vx0': 103.67629902591716, 'vy0': 725.7214163473893, 'vz0': -67.99194802672406, 'm0': 75000.0}

        # Scaling factors for non-dimensionalization
        scale_factors = {
            'x': planet_params['R'],
            'y': planet_params['R'],
            'z': planet_params['R'],
            'vx': 1000.0,
            'vy': 1000.0,
            'vz': 1000.0,
            'm': rocket_params['mass_start'],
            'ux': rocket_params['thrust_max'],
            'uy': rocket_params['thrust_max'],
            'uz': rocket_params['thrust_max'],
            'gamma': rocket_params['thrust_max'],
            't': bounds['t_max']
        }

        # Offsets for non-dimensionalization
        offsets = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'm': 0.0,
            'ux': 0.0,
            'uy': 0.0,
            'uz': 0.0,
            'gamma': 0.0,
            't': 0.0
        }

        print(f"Initial conditions (run {i+1}): {initial_conditions}")

        result = run_optimization(initial_conditions, terminal_conditions, planet_params, rocket_params, bounds, landing_tolerances, scale_factors, offsets, N)
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
        traj_plot.plot_single_run(successful_runs[0], N, planet_params['R'])
        comparison = traj_val.compare_optimized_vs_simulated_trajectories(dynamics_np, initial_conditions, successful_runs[0], planet_params, rocket_params, N)
        traj_val.plot_trajectory_comparison(successful_runs[0], comparison, N)
    elif (num_runs > 1 and num_success > 0):
        traj_plot.plot_multiple_runs(successful_runs)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    main()