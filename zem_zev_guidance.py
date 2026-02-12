import numpy as np
from scipy.optimize import minimize_scalar

import trajectory_plotting as traj_plot
import trajectory_validation as traj_val

# TODO: ADD JIT
# TODO: Share dynamics across different trajectory optimization methods
def state_dynamics(state, control, planet_params, rocket_params):
    """
    Computes the time derivative of the state vector for a rocket under the influence of gravity, drag, and thrust.

    Args:
        state (np.array): Current state vector [x, y, z, vx, vy, vz, m]
        control (np.array): Control input vector [ux, uy, uz]
        planet_params (dict): Physical parameters
        rocket_params (dict): Rocket parameters

    Returns:
        state_dot (np.array): Time derivative of the state vector
    """
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

    r = np.sqrt(x**2 + y**2 + z**2)
    u_total = np.sqrt(ux**2 + uy**2 + uz**2)

    r = np.sqrt(x**2 + y**2 + z**2)
    h = r - R
    rho = rho0 * np.exp(-h / H)
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    D = -0.5 * rho * Cd * Aref * v * np.array([vx, vy, vz])
    Dx = D[0]
    Dy = D[1]
    Dz = D[2]

    # TODO: Neaten up the dynamics function into vectors for xdot, vdot, and mdot
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -G * M * (x / r**3) + Dx / m + ux / m
    dvydt = -G * M * (y / r**3) + Dy / m + uy / m
    dvzdt = -G * M * (z / r**3) + Dz / m + uz / m
    dmdt = -u_total / (Isp * g0)

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dmdt])

# TODO: ADD JIT
def rk4_step(dynamics_fn, state, control, planet_params, rocket_params, dt):
    """
    Perform a single RK4 integration step.

    Args:
        dynamics_fn (function): Function to compute state derivatives
        state (np.array): Current state vector
        control (np.array): Control input vector
        planet_params (dict): Physical parameters
        rocket_params (dict): Rocket parameters
        dt (float): Time step (s)

    Returns:
        next_state (np.array): State vector after time step dt
    """

    k1 = dynamics_fn(state, control, planet_params, rocket_params)
    k2 = dynamics_fn(state + 0.5 * dt * k1, control, planet_params, rocket_params)
    k3 = dynamics_fn(state + 0.5 * dt * k2, control, planet_params, rocket_params)
    k4 = dynamics_fn(state + dt * k3, control, planet_params, rocket_params)

    next_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return next_state

# TODO: ADD JIT
def propagate_states_rk4(dynamics_fn, initial_state, controls, planet_params, rocket_params, tf, N):
    """
    Simulate the trajectory using RK4 integration with control inputs.

    Args:
        dynamics_fn (function): Function to compute state derivatives
        initial_state (np.array): Array of system states at initial time
        controls (np.array): Array of control inputs at each timenode
        planet_params (dict): Physical parameters
        rocket_params (dict): Rocket parameters
        tf (float): Final time (s)
        N (int): Number of integration steps

    Returns:
        states (np.array): (num_states, N+1) array of propagated states
    """

    dt = tf / N
    num_states = initial_state.shape[0]

    states = np.zeros((num_states, N+1))
    states[:, 0] = initial_state

    for k in range(N):
        states[:, k+1] = rk4_step(dynamics_fn, states[:, k], controls[:, k], planet_params, rocket_params, dt)

    return states

# TODO: ADD JIT
def zem_zev_guidance(dynamics_fn, states, r_target, v_target, t_go, planet_params, rocket_params):
    """
    Calculates the commanded acceleration using the ZEM/ZEV feedback law.

    Args:
        dynamics_fn (function): Function to compute state derivatives
        states (np.array): Current state vector [x, y, z, vx, vy, vz, m]
        r_target (np.array): Desired final position (m)
        v_target (np.array): Desired final velocity (m/s)
        t_go (float): Time-to-go (s)
        planet_params (dict): Physical parameters
        rocket_params (dict): Rocket parameters

    Returns:
        a_cmd (np.array): Commanded acceleration vector (m/s^2)
    """

    # Calculate Zero-Effort-Miss (ZEM) and Zero-Effort-Velocity (ZEV) based on the predicted final state
    N = 10 # Number of steps to propagate with zero control for ZEM/ZEV calculation
    zero_controls = np.zeros((3, N+1)) # Zero control inputs for propagation
    X = propagate_states_rk4(dynamics_fn, states, zero_controls, planet_params, rocket_params, t_go, N)
    r_pred = X[0:3, -1]  # Predicted final position after propagating with zero control
    v_pred = X[3:6, -1]  # Predicted final velocity after propagating with zero control

    zem = r_target - r_pred
    zev = v_target - v_pred

    # Apply control law
    a_cmd = (6.0 / t_go**2) * zem - (2.0 / t_go) * zev

    return a_cmd

def saturate_thrust_accel(a_cmd, m, T_min, T_max):
    """
    Saturates the commanded acceleration based on the minimum and maximum thrust limits.

    Args:
        a_cmd (np.array): Commanded acceleration vector (m/s^2)
        m (float): Current mass of the vehicle (kg)
        T_min (float): Minimum thrust (N)
        T_max (float): Maximum thrust (N)

    Returns:
        a_cmd (np.array): Saturated acceleration vector (m/s^2)
    """
   # T_max/m is the maximum acceleration magnitude
   # a_sat = {a                 if |a| <= T_max/m and |a| >= T_min/m,
   #         {a * (T_min/m)/|a| if |a| < T_min/m,
   #         {a * (T_max/m)/|a| if |a| > T_max/m

    a_min = T_min / m
    a_max = T_max / m

    a_norm = np.linalg.norm(a_cmd)
    if a_norm < 1e-8:
        return np.zeros_like(a_cmd)

    # Note: Using a lower bound can cause "chattering" in control if the commanded acceleration stays around the minimum threshold.
    if a_norm < a_min:
        a_sat = a_cmd * (a_min / a_norm)
        return a_sat
    elif a_norm > a_max:
        a_sat = a_cmd * (a_max / a_norm)
        return a_sat
    else:
        return a_cmd

def run_simulation(initial_conditions, terminal_conditions, planet_params, rocket_params, N, t_final):
    """
    Runs the trajectory simulation for a specific flight time.
    """

    dt = t_final / N  # Simulation time step (s)

    # Vehicle Parameters
    T_min = rocket_params['thrust_min'] # Minimum thrust (N)
    T_max = rocket_params['thrust_max'] # Maximum thrust (N)

    # Storage
    t_hist = []
    r_hist = []
    v_hist = []
    a_cmd_hist = []
    m_hist = []

    # Initial State
    t = 0
    r = np.array([initial_conditions['x0'], initial_conditions['y0'], initial_conditions['z0']])
    v = np.array([initial_conditions['vx0'], initial_conditions['vy0'], initial_conditions['vz0']])
    m = initial_conditions['m0']

    # Target State
    r_f = np.array([terminal_conditions['xf'], terminal_conditions['yf'], terminal_conditions['zf']])
    v_f = np.array([terminal_conditions['vxf'], terminal_conditions['vyf'], terminal_conditions['vzf']])

    # Compute/store node 0 (t=0)
    t_go = t_final - t
    states = np.concatenate((r, v, [m]))
    a_cmd = zem_zev_guidance(state_dynamics, states, r_f, v_f, t_go, planet_params, rocket_params)
    a_cmd = saturate_thrust_accel(a_cmd, m, T_min, T_max)

    t_hist.append(t)
    r_hist.append(r.copy())
    v_hist.append(v.copy())
    a_cmd_hist.append(a_cmd.copy())
    m_hist.append(m)

    for k in range(N):
        t_go = t_final - t

        # Guard against numerical issues at end of flight
        if t_go < 0.001:
            break

        # Control law
        states = np.concatenate((r, v, [m]))
        a_cmd = zem_zev_guidance(state_dynamics, states, r_f, v_f, t_go, planet_params, rocket_params)
        a_cmd = saturate_thrust_accel(a_cmd, m, T_min, T_max)

        # State propagation
        X = rk4_step(state_dynamics, states, m*a_cmd, planet_params, rocket_params, dt)

        r = X[0:3]
        v = X[3:6]
        m = X[6]
        t = t + dt

        # Store Data
        t_hist.append(t)
        r_hist.append(r.copy())
        v_hist.append(v.copy())
        a_cmd_hist.append(a_cmd.copy())
        m_hist.append(m)

    # Final error check
    pos_error = np.linalg.norm(r - r_f)
    vel_error = np.linalg.norm(v - v_f)
    print(f"Final position error: {pos_error:.2f} m")
    print(f"Final velocity error: {vel_error:.2f} m/s")

    r_hist = np.asarray(r_hist, dtype=float)
    v_hist = np.asarray(v_hist, dtype=float)
    m_hist = np.asarray(m_hist, dtype=float)
    a_cmd_hist = np.asarray(a_cmd_hist, dtype=float)
    ux_hist = a_cmd_hist[:,0] * m_hist
    uy_hist = a_cmd_hist[:,1] * m_hist
    uz_hist = a_cmd_hist[:,2] * m_hist

    return {
        "success": True,
        "iterations": 0,
        "solve_time": 0,
        "tf_opt": t_final,
        "x_opt": r_hist[:,0],
        "y_opt": r_hist[:,1],
        "z_opt": r_hist[:,2],
        "vx_opt": v_hist[:,0],
        "vy_opt": v_hist[:,1],
        "vz_opt": v_hist[:,2],
        "m_opt": m_hist,
        "ux_opt": ux_hist,
        "uy_opt": uy_hist,
        "uz_opt": uz_hist,
        "gamma_opt": None
    }

def find_optimal_time(initial_conditions, terminal_conditions, planet_params, rocket_params, N, t_min, t_max):
    print("\n--- Performing line search for optimal time of flight ---")

    def cost_function(t_final):
        result = run_simulation(initial_conditions, terminal_conditions, planet_params, rocket_params, N, t_final)
        final_mass = result['m_opt'][-1]
        return -final_mass  # maximize final mass (minimize negative final mass)

    result = minimize_scalar(cost_function, method='bounded', bounds=(t_min, t_max), options={'xatol': 1.0, 'disp': 3})
    t_opt = result.x
    cost = result.fun

    print(f"Optimal time of flight: {t_opt:.2f} s")
    print(f"Cost function value: {cost:.2f} kg")

    return t_opt

def run_optimization(initial_conditions, terminal_conditions, planet_params, rocket_params, N_opt, N_final, t_min, t_max):
    # Optimize
    t_opt = find_optimal_time(initial_conditions, terminal_conditions, planet_params, rocket_params, N_opt, t_min, t_max)

    # Run final simulation using the optimal time of flight found from the line search
    result = run_simulation(initial_conditions, terminal_conditions, planet_params, rocket_params, N_final, t_opt)

    return result

def main():
    # TODO: Share parameters and initial conditions across different trajectory optimization methods for easier comparison

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
        'g0': 9.81, # Standard gravity (m/s^2)
        'mass_start': 75000.0, # Starting mass (kg) - Falcon 9 Mass at start of boostback burn
        'mass_dry': 22200.0, # Dry mass (kg) - Falcon 9 Dry Mass
        'Aref': np.pi*(3.7/2)**2, # Reference area (m^2)
        'Cd': 1.8 # Drag coefficient (assumed)
    }

    terminal_conditions = {
        'xf': 0.0, # Final x-position (m)
        'yf': 0.0, # Final y-position (m)
        'zf': planet_params['R'], # Final z-position (m) - on the surface
        'vxf': 0.0, # Final x-velocity (m/s)
        'vyf': 0.0, # Final y-velocity (m/s)
        'vzf': 0.0, # Final z-velocity (m/s)
        'mf': rocket_params['mass_dry'] # Minimum ALLOWABLE final mass (kg)
    }

    num_runs = 1
    N_opt = 50 # Number of simulation steps
    N_final = 10*N_opt # Number of steps for final simulation (more steps for smoother plots)
    t_min = 10.0 # Minimum time of flight (s)
    t_max = 500.0 # Maximum time of flight (s)

    np.random.seed(8)
    results = []
    for i in range(num_runs):
        # Generate random initial conditions for each run
        initial_conditions = {
            'x0': np.random.uniform(-50000.0, 50000.0), # m
            'y0': np.random.uniform(-50000.0, 50000.0), # m
            'z0': planet_params['R'] + np.random.uniform(50000.0, 150000.0), # m
            'vx0': np.random.uniform(-1000.0, 1000.0), # m/s
            'vy0': np.random.uniform(-1000.0, 1000.0), # m/s
            'vz0': np.random.uniform(-500.0, 500.0), # m/s
            'm0': rocket_params['mass_start'] # kg
        }

        # initial_conditions = {'x0': 10333.00064753613, 'y0': -14102.69257401282, 'z0': 6451802.3336159075 , 'vx0': 646.6701807114714, 'vy0': 121.62380764174964, 'vz0': -152.95896172646286, 'm0': 75000.0}
        # initial_conditions = {'x0': -10988.148096549397, 'y0': 39553.273620720895, 'z0': 6430847.15719154, 'vx0': 821.1205393090227, 'vy0': 457.70904823551905, 'vz0': -372.8149990047769, 'm0': 75000.0}
        # initial_conditions = {'x0': -2229.5538019762243, 'y0': 25230.130182274952, 'z0': 6450341.862722686, 'vx0': 15.920842801517665, 'vy0': 454.84039373955056, 'vz0': -156.4300590392359, 'm0': 75000.0}
        # initial_conditions = {'x0': -20476.74314880714, 'y0': -13615.78880679213, 'z0': 6444010.141756712, 'vx0': 392.54362697458873, 'vy0': 284.2714639006099, 'vz0': -267.7434706956784, 'm0': 75000.0}
        # initial_conditions = {'x0': 28840.55161689737, 'y0': -20875.4851646929, 'z0': 6434001.974537622, 'vx0': 559.796802578966, 'vy0': 368.6994615807289, 'vz0': -204.34760760308177, 'm0': 75000.0}
        # initial_conditions = {'x0': -31518.45806321033, 'y0': 47770.549924390696, 'z0': 6433824.722904884, 'vx0': 209.2517851410796, 'vy0': 515.7533383325085, 'vz0': -79.33290732940077, 'm0': 75000.0}
        # initial_conditions = {'x0': 31116.18834065397, 'y0': 28482.906354009217, 'z0': 6449458.003926393, 'vx0': 634.4300909114133, 'vy0': 76.80522338545347, 'vz0': -192.1010130966052, 'm0': 75000.0}
        # initial_conditions = {'x0': -1481.8618443154555, 'y0': 28626.116533809356, 'z0': 6437017.691213863, 'vx0': 103.67629902591716, 'vy0': 725.7214163473893, 'vz0': -67.99194802672406, 'm0': 75000.0}

        result = run_optimization(initial_conditions, terminal_conditions, planet_params, rocket_params, N_opt, N_final, t_min, t_max)
        results.append(result)

    if (num_runs == 1):
        traj_plot.plot_single_run(results[0], N_final, planet_params['R'])
        # comparison = traj_val.compare_optimized_vs_simulated_trajectories(dynamics_np, initial_conditions, results[0], planet_params, rocket_params, N_opt)
        # traj_val.plot_trajectory_comparison(results[0], comparison, N_opt)
    if (num_runs > 1):
        traj_plot.plot_multiple_runs(results)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    main()