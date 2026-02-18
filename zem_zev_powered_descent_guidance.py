import numpy as np
from scipy.optimize import minimize_scalar
from time import time

import trajectory_plotting as traj_plot
import trajectory_validation as traj_val

class ZEMZEVPoweredDescentGuidance:
    def __init__(self, planet_params: dict, rocket_params: dict, state_dynamics_func: callable):
        """
        Initializes the ZEM/ZEV trajectory generator. For more information, see the paper titled "Applications of Generalized Zero-Effort-Miss/Zero-Effort-Velocity Feedback Guidance Algorithm" by Guo, Hawkins and Wie.

        Args:
            planet_params (dict): Dictionary containing planetary parameters (e.g., gravity, atmosphere)
            rocket_params (dict): Dictionary containing rocket parameters (e.g., mass, thrust limits)
            state_dynamics_func (callable): Function that computes the time derivative of the state vector given the current state and control input
        """

        self.planet_params = planet_params
        self.rocket_params = rocket_params
        self.state_dynamics_func = state_dynamics_func

    # TODO: ADD JIT
    def _rk4_step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform a single RK4 integration step.

        Args:
            state (np.ndarray): Current state vector
            control (np.ndarray): Control input vector
            dt (float): Time step (s)

        Returns:
            next_state (np.ndarray): State vector after time step dt
        """

        k1 = self.state_dynamics_func(state, control, self.planet_params, self.rocket_params)
        k2 = self.state_dynamics_func(state + 0.5 * dt * k1, control, self.planet_params, self.rocket_params)
        k3 = self.state_dynamics_func(state + 0.5 * dt * k2, control, self.planet_params, self.rocket_params)
        k4 = self.state_dynamics_func(state + dt * k3, control, self.planet_params, self.rocket_params)

        next_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return next_state

    # TODO: ADD JIT
    def _propagate_states_rk4(self, initial_state: np.ndarray, controls: np.ndarray, tf: float, N: int) -> np.ndarray:
        """
        Simulate the trajectory using RK4 integration with control inputs.

        Args:
            initial_state (np.ndarray): Array of system states at initial time
            controls (np.ndarray): Array of control inputs at each timenode
            tf (float): Final time (s)
            N (int): Number of integration steps

        Returns:
            states (np.ndarray): (num_states, N+1) array of propagated states
        """

        dt = tf / N
        num_states = initial_state.shape[0]

        states = np.zeros((num_states, N+1))
        states[:, 0] = initial_state

        for k in range(N):
            states[:, k+1] = self._rk4_step(states[:, k], controls[:, k], dt)

        return states

    # TODO: ADD JIT
    def _compute_accel_command(self, state: np.ndarray, r_target: np.ndarray, v_target: np.ndarray, t_go: float, N_pred: int) -> np.ndarray:
        """
        Calculates the commanded acceleration using the ZEM/ZEV feedback law.

        Args:
            state (np.ndarray): Current state vector [x, y, z, vx, vy, vz, m]
            r_target (np.ndarray): Desired final position (m)
            v_target (np.ndarray): Desired final velocity (m/s)
            t_go (float): Time-to-go (s)
            N_pred (int): Number of integration steps for ZEM/ZEV prediction

        Returns:
            a_cmd (np.ndarray): Commanded acceleration vector (m/s^2)
        """

        # Predict terminal state under zero thrust
        zero_controls = np.zeros((3, N_pred+1)) # Zero control inputs for propagation
        X_pred = self._propagate_states_rk4(state, zero_controls, t_go, N_pred)
        r_pred = X_pred[0:3, -1]  # Predicted final position after propagating with zero control
        v_pred = X_pred[3:6, -1]  # Predicted final velocity after propagating with zero control

        # Calculate Zero-Effort-Miss (ZEM) and Zero-Effort-Velocity (ZEV) based on the predicted final state
        zem = r_target - r_pred
        zev = v_target - v_pred

        # Apply control law
        Kp = 6.0 / t_go**2
        Kv = 2.0 / t_go
        a_cmd = Kp * zem - Kv * zev

        # print(f"ZEM: {zem}, ZEV: {zev}, Kp: {Kp:.2f}, Kv: {Kv:.2f}")

        return a_cmd

    # TODO: ADD JIT
    def _saturate_accel_command(self, a_cmd: np.ndarray, m: float) -> np.ndarray:
        """
        Saturates the commanded acceleration based on the minimum and maximum thrust limits.

        a_sat = {a                 if |a| <= T_max/m and |a| >= T_min/m,
                {a * (T_min/m)/|a| if |a| < T_min/m,
                {a * (T_max/m)/|a| if |a| > T_max/m

        Args:
            a_cmd (np.ndarray): Commanded acceleration vector (m/s^2)
            m (float): Current mass of the vehicle (kg)

        Returns:
            a_cmd (np.ndarray): Saturated acceleration vector (m/s^2)
        """

        a_min = self.rocket_params['thrust_min'] / m
        a_max = self.rocket_params['thrust_max'] / m

        a_norm = np.linalg.norm(a_cmd)
        if a_norm < 1e-6:
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

    # TODO: ADD JIT
    def _generate_trajectory(self, initial_conditions: dict, terminal_conditions: dict, t_final: float, N: int, N_pred: int, verbose: bool = False) -> dict:
        """
        Generates a trajectory using the ZEM/ZEV guidance law.

        Args:
            initial_conditions (dict): Dictionary containing initial state parameters
            terminal_conditions (dict): Dictionary containing terminal state parameters
            t_final (float): Time of flight for the trajectory (s)
            N (int): Number of integration steps for trajectory generation
            N_pred (int): Number of integration steps for ZEM/ZEV prediction
            verbose (bool): If True, prints trajectory generation final errors

        Returns:
            trajectory (dict): Dictionary containing trajectory data for plotting and analysis
        """

        # Storage
        r_hist = []
        v_hist = []
        m_hist = []
        u_hist = []
        t_hist = []

        # Initial State
        r = np.array([initial_conditions['x'], initial_conditions['y'], initial_conditions['z']])
        v = np.array([initial_conditions['vx'], initial_conditions['vy'], initial_conditions['vz']])
        m = initial_conditions['m']
        t = 0.0

        # Target State
        r_target = np.array([terminal_conditions['x'], terminal_conditions['y'], terminal_conditions['z']])
        v_target = np.array([terminal_conditions['vx'], terminal_conditions['vy'], terminal_conditions['vz']])

        # Compute/store node 0 (t=0)
        t_go = t_final - t
        states = np.concatenate((r, v, [m]))
        a_cmd = self._compute_accel_command(states, r_target, v_target, t_go, N_pred)
        a_cmd = self._saturate_accel_command(a_cmd, m)
        u = m * a_cmd

        r_hist.append(r.copy())
        v_hist.append(v.copy())
        m_hist.append(m)
        u_hist.append(u.copy())
        t_hist.append(t)

        # Integration time step (s)
        dt = t_final / N

        for k in range(N):
            t_go = t_final - t

            # TODO: Find a better way to handle the end of flight condition. Right now we just break the loop when t_go gets very small
            # Guard against numerical issues at end of flight
            if t_go < 0.001:
                break

            # Control law
            state = np.concatenate((r, v, [m]))
            a_cmd = self._compute_accel_command(state, r_target, v_target, t_go, N_pred)
            a_cmd = self._saturate_accel_command(a_cmd, m)
            u = m * a_cmd

            # State propagation
            X = self._rk4_step(state, u, dt)
            r = X[0:3]
            v = X[3:6]
            m = X[6]
            t = t + dt

            r_hist.append(r.copy())
            v_hist.append(v.copy())
            m_hist.append(m)
            u_hist.append(u.copy())
            t_hist.append(t)

        r_hist = np.asarray(r_hist, dtype=float)
        v_hist = np.asarray(v_hist, dtype=float)
        m_hist = np.asarray(m_hist, dtype=float)
        u_hist = np.asarray(u_hist, dtype=float)
        t_hist = np.asarray(t_hist, dtype=float)

        # Final error check
        pos_error = np.linalg.norm(r_hist[-1] - r_target)
        vel_error = np.linalg.norm(v_hist[-1] - v_target)

        if verbose:
            print(f"Final position error: {pos_error:.2f} m")
            print(f"Final velocity error: {vel_error:.2f} m/s")

        return {
            "r": r_hist,
            "v": v_hist,
            "m": m_hist,
            "u": u_hist,
            "t": t_hist
        }

    def find_best_trajectory(self, initial_conditions: dict, terminal_conditions: dict, t_min: float, t_max: float, t_tol: float, r_tol: float, v_tol: float, N: int, N_final: int, N_pred: int, verbose: bool = False) -> dict:
        """
        Performs a line search over the time of flight to find the trajectory that maximizes final mass (minimizes fuel consumption).

        Args:
            initial_conditions (dict): Dictionary containing initial state parameters
            terminal_conditions (dict): Dictionary containing terminal state parameters
            t_min (float): Minimum time of flight for line search
            t_max (float): Maximum time of flight for line search
            t_tol (float): Tolerance for time of flight optimization
            r_tol (float): Position tolerance for successful trajectory (m)
            v_tol (float): Velocity tolerance for successful trajectory (m/s)
            N (int): Number of integration steps for trajectory generation during line search
            N_final (int): Number of integration steps for final trajectory generation (more steps for more accurate results)
            N_pred (int): Number of integration steps for ZEM/ZEV prediction
            verbose (bool): If True, prints optimization results and trajectory errors

        Returns:
            result (dict): Dictionary containing optimization results and best trajectory data
        """

        def trajectory_cost(t_final: float) -> float:
            trajectory = self._generate_trajectory(initial_conditions, terminal_conditions, t_final, N, N_pred, verbose)
            cost = -trajectory['m'][-1]
            return cost  # Minimize negative final mass (maximize final mass)

        disp = 3 if verbose else 0
        t_start = time()
        opt_result = minimize_scalar(trajectory_cost, method='bounded', bounds=(t_min, t_max), options={'xatol': t_tol, 'disp': disp})
        t_end = time()
        t_solve = t_end - t_start
        t_opt = opt_result.x
        cost = opt_result.fun

        if verbose:
            print(f"Optimal time of flight: {t_opt:.2f} s")
            print(f"Cost function value: {cost:.2f} kg")

        # Run final trajectory generation using the time of flight that produced the lowest cost trajectory
        best_trajectory = self._generate_trajectory(initial_conditions, terminal_conditions, t_opt, N_final, N_pred, verbose)

        best_rf = np.array([best_trajectory['r'][-1,0], best_trajectory['r'][-1,1], best_trajectory['r'][-1,2]])
        best_vf = np.array([best_trajectory['v'][-1,0], best_trajectory['v'][-1,1], best_trajectory['v'][-1,2]])
        target_rf = np.array([terminal_conditions['x'], terminal_conditions['y'], terminal_conditions['z']])
        target_vf = np.array([terminal_conditions['vx'], terminal_conditions['vy'], terminal_conditions['vz']])
        pos_error = np.linalg.norm(best_rf - target_rf)
        vel_error = np.linalg.norm(best_vf - target_vf)
        if verbose:
            print(f"Best trajectory final position error: {pos_error:.2f} m")
            print(f"Best trajectory final velocity error: {vel_error:.2f} m/s")
            print(f"Best trajectory final mass: {best_trajectory['m'][-1]:.2f} kg")

        success = True
        if (pos_error > r_tol or vel_error > v_tol):
            success = False
            print("WARNING: Best trajectory failed to meet terminal conditions within acceptable error thresholds. Final position error: {:.2f} m, final velocity error: {:.2f} m/s".format(pos_error, vel_error))
        # if best_trajectory['m'][-1] < terminal_conditions['mf']:
        #     best_trajectory['success'] = False
        #     print("WARNING: Best trajectory failed to meet minimum final mass requirement. Final mass: {:.2f} kg, minimum required mass: {:.2f} kg".format(best_trajectory['m'][-1], terminal_conditions['mf']))

        return {
            "success": success,
            "iterations": opt_result.nit,
            "solve_time": t_solve,
            "tf_opt": t_opt,
            "x_opt": best_trajectory['r'][:,0],
            "y_opt": best_trajectory['r'][:,1],
            "z_opt": best_trajectory['r'][:,2],
            "vx_opt": best_trajectory['v'][:,0],
            "vy_opt": best_trajectory['v'][:,1],
            "vz_opt": best_trajectory['v'][:,2],
            "m_opt": best_trajectory['m'],
            "ux_opt": best_trajectory['u'][:,0],
            "uy_opt": best_trajectory['u'][:,1],
            "uz_opt": best_trajectory['u'][:,2],
            "gamma_opt": None
        }

# TODO: ADD JIT
# TODO: Share dynamics across different trajectory optimization methods
def state_dynamics(state: np.array, control: np.array, planet_params: dict, rocket_params: dict) -> np.array:
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
        'Aref': np.pi*(3.7/2.0)**2, # Reference area (m^2)
        'Cd': 1.8 # Drag coefficient (assumed)
    }

    terminal_conditions = {
        'x': 0.0, # Final x-position (m)
        'y': 0.0, # Final y-position (m)
        'z': planet_params['R'], # Final z-position (m) - on the surface
        'vx': 0.0, # Final x-velocity (m/s)
        'vy': 0.0, # Final y-velocity (m/s)
        'vz': 0.0, # Final z-velocity (m/s)
        'm': rocket_params['mass_dry'] # Minimum ALLOWABLE final mass (kg)
    }

    num_runs = 1
    N = 50 # Number of integration steps in generated trajectories during line search
    N_final = 10*N # Number of integration steps in final optimized trajectory
    N_pred = 10 # Number of integration steps for ZEM/ZEV prediction
    t_min = 10.0 # Minimum time of flight in line search (s)
    t_max = 500.0 # Maximum time of flight in line search (s)
    t_tol = 1.0 # Optimal time of flight tolerance in line search (s)
    r_tol = 10.0 # Position tolerance for successful trajectory (m)
    v_tol = 10.0 # Velocity tolerance for successful trajectory (m/s)
    verbose = True

    results = []
    failed_initial_conditions = []
    np.random.seed(8)
    for i in range(num_runs):
        # Generate random initial conditions for each run
        initial_conditions = {
            'x': np.random.uniform(-50000.0, 50000.0), # m
            'y': np.random.uniform(-50000.0, 50000.0), # m
            'z': planet_params['R'] + np.random.uniform(50000.0, 150000.0), # m
            'vx': np.random.uniform(-1000.0, 1000.0), # m/s
            'vy': np.random.uniform(-1000.0, 1000.0), # m/s
            'vz': np.random.uniform(-500.0, 500.0), # m/s
            'm': rocket_params['mass_start'] # kg
        }

        # initial_conditions = {'x0': 48045.32429793403, 'y0': 38719.85545747625, 'z0': 6480179.72465158, 'vx0': 951.9382652644483, 'vy0': 968.2924888876025, 'vz0': -271.9402464272812, 'm0': 75000.0}

        print(f"Initial conditions (run {i+1}): {initial_conditions}")

        zem_zev_pdg = ZEMZEVPoweredDescentGuidance(planet_params, rocket_params, state_dynamics)
        result = zem_zev_pdg.find_best_trajectory(initial_conditions, terminal_conditions, t_min, t_max, t_tol, r_tol, v_tol, N, N_final, N_pred, verbose)
        results.append(result)
        if not result['success']:
            failed_initial_conditions.append(initial_conditions)

    successful_runs = [r for r in results if r["success"]]
    num_success = len(successful_runs)

    if failed_initial_conditions:
        print("\n--- Failed Initial Conditions ---")
        for initial_condition in failed_initial_conditions:
            print(initial_condition)

    if (num_runs == 1):
        traj_plot.plot_single_run(successful_runs[0], N_final, planet_params['R'])
        # comparison = traj_val.compare_optimized_vs_simulated_trajectories(dynamics_np, initial_conditions, successful_runs[0], planet_params, rocket_params, N_final)
        # traj_val.plot_trajectory_comparison(successful_runs[0], comparison, N_final)
    if (num_runs > 1 and num_success > 0):
        traj_plot.plot_multiple_runs(successful_runs)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    main()