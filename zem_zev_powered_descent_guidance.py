from enum import IntEnum

import numpy as np
from scipy.optimize import minimize_scalar
from time import time

import trajectory_plotting as traj_plot
import trajectory_validation as traj_val
from common import State, Control

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

        if state_dynamics_func is None:
            print("Using dynamics function internal to class.")
            self.state_dynamics_func = self.dynamics
        else:
            print("Using user-provided dynamics function.")
            self.state_dynamics_func = state_dynamics_func

    def dynamics(self, state, control, planet_params: dict, rocket_params: dict):
        """
        State dynamics implementation for a 3DOF powered descent guidance problem.
        state = [x, y, z, vx, vy, vz, m]
        control = [ux, uy, uz] or [ux, uy, uz, gamma]
        returns derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dm/dt]
        """

        r = state[State.X:State.Z+1]
        v = state[State.VX:State.VZ+1]
        m = state[State.M]
        u = control[Control.X:Control.Z+1]

        G = planet_params['G']
        M = planet_params['M']
        R = planet_params['R']
        rho0 = planet_params['rho0']
        H = planet_params['H']
        Isp = rocket_params['Isp']
        g0 = rocket_params['g0']
        Aref = rocket_params['Aref']
        Cd = rocket_params['Cd']

        eps = 1.0e-8
        r_mag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        v_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2 + eps)
        rho = rho0 * np.exp(-(r_mag - R) / H)
        D = -0.5 * rho * Cd * Aref * v_mag * v

        u_mag = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2 + eps)

        drdt = v
        dvdt = -G * M * (r / r_mag**3) + u / m + D / m
        dmdt = -u_mag / (Isp * g0)

        return np.concatenate([drdt, dvdt, [dmdt]])

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
        r_pred = X_pred[State.X:State.Z+1, -1]  # Predicted final position after propagating with zero control
        v_pred = X_pred[State.VX:State.VZ+1, -1]  # Predicted final velocity after propagating with zero control

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
    def _generate_trajectory(self, initial_conditions: np.ndarray, terminal_conditions: np.ndarray, t_final: float, N: int, N_pred: int, verbose: bool = False) -> dict:
        """
        Generates a trajectory using the ZEM/ZEV guidance law.

        Args:
            initial_conditions (np.ndarray): Array containing initial state parameters
            terminal_conditions (np.ndarray): Array containing terminal state parameters
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
        r = np.array([initial_conditions[State.X], initial_conditions[State.Y], initial_conditions[State.Z]])
        v = np.array([initial_conditions[State.VX], initial_conditions[State.VY], initial_conditions[State.VZ]])
        m = initial_conditions[State.M]
        t = 0.0

        # Target State
        r_target = np.array([terminal_conditions[State.X], terminal_conditions[State.Y], terminal_conditions[State.Z]])
        v_target = np.array([terminal_conditions[State.VX], terminal_conditions[State.VY], terminal_conditions[State.VZ]])

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
            r = X[State.X:State.Z+1]
            v = X[State.VX:State.VZ+1]
            m = X[State.M]
            t = t + dt

            r_hist.append(r.copy())
            v_hist.append(v.copy())
            m_hist.append(m)
            u_hist.append(u.copy())
            t_hist.append(t)

        r_hist = np.asarray(r_hist, dtype=float).T
        v_hist = np.asarray(v_hist, dtype=float).T
        m_hist = np.asarray(m_hist, dtype=float)
        u_hist = np.asarray(u_hist, dtype=float).T
        t_hist = np.asarray(t_hist, dtype=float)

        # Final error check
        pos_error = np.linalg.norm(r_hist[:, -1] - r_target)
        vel_error = np.linalg.norm(v_hist[:, -1] - v_target)

        if verbose:
            print(f"Final position error: {pos_error:.2f} m")
            print(f"Final velocity error: {vel_error:.2f} m/s")

        return {
            't': t_hist,
            'X': np.vstack((r_hist, v_hist, m_hist)),
            'U': u_hist,
        }

    def find_best_trajectory(self, initial_conditions: np.ndarray, terminal_conditions: np.ndarray, t_min: float, t_max: float, t_tol: float, r_tol: float, v_tol: float, N: int, N_final: int, N_pred: int, verbose: bool = False) -> dict:
        """
        Performs a line search over the time of flight to find the trajectory that maximizes final mass (minimizes fuel consumption).

        Args:
            initial_conditions (np.ndarray): Array containing initial state parameters
            terminal_conditions (np.ndarray): Array containing terminal state parameters
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
            cost = -trajectory['X'][State.M, -1]  # Final mass is the last element in the state vector, negate for minimization
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

        best_rf = np.array([best_trajectory['X'][State.X, -1], best_trajectory['X'][State.Y, -1], best_trajectory['X'][State.Z, -1]])
        best_vf = np.array([best_trajectory['X'][State.VX, -1], best_trajectory['X'][State.VY, -1], best_trajectory['X'][State.VZ, -1]])
        target_rf = np.array([terminal_conditions[State.X], terminal_conditions[State.Y], terminal_conditions[State.Z]])
        target_vf = np.array([terminal_conditions[State.VX], terminal_conditions[State.VY], terminal_conditions[State.VZ]])
        pos_error = np.linalg.norm(best_rf - target_rf)
        vel_error = np.linalg.norm(best_vf - target_vf)
        if verbose:
            print(f"Best trajectory final position error: {pos_error:.2f} m")
            print(f"Best trajectory final velocity error: {vel_error:.2f} m/s")
            print(f"Best trajectory final mass: {best_trajectory['X'][State.M, -1]:.2f} kg")

        success = True
        if (pos_error > r_tol or vel_error > v_tol):
            success = False
            print("WARNING: Best trajectory failed to meet terminal conditions within acceptable error thresholds. Final position error: {:.2f} m, final velocity error: {:.2f} m/s".format(pos_error, vel_error))
        # if best_trajectory['X'][State.M, -1] < terminal_conditions[State.M]:
        #     best_trajectory['success'] = False
        #     print("WARNING: Best trajectory failed to meet minimum final mass requirement. Final mass: {:.2f} kg, minimum required mass: {:.2f} kg".format(best_trajectory['X'][State.M, -1], terminal_conditions[State.M]))

        return {
            'success': success,
            'iterations': opt_result.nit,
            'solve_time': t_solve,
            'tf': t_opt,
            'X': best_trajectory['X'],
            'U': best_trajectory['U'],
        }

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

    terminal_conditions = np.zeros(7)
    terminal_conditions[State.X] = 0.0
    terminal_conditions[State.Y] = 0.0
    terminal_conditions[State.Z] = planet_params['R']
    terminal_conditions[State.VX] = 0.0
    terminal_conditions[State.VY] = 0.0
    terminal_conditions[State.VZ] = 0.0
    terminal_conditions[State.M] = rocket_params['mass_dry']

    num_runs = 100
    N = 50 # Number of integration steps in generated trajectories during line search
    N_final = 10*N # Number of integration steps in final optimized trajectory
    # NOTE: The choice of N_pred can have a significant impact on the generated control commands (e.g., lower N_pred can cause more jitter)
    N_pred = 10 # Number of integration steps for ZEM/ZEV prediction
    t_min = 10.0 # Minimum time of flight in line search (s)
    t_max = 500.0 # Maximum time of flight in line search (s)
    t_tol = 1.0 # Optimal time of flight tolerance in line search (s)
    r_tol = 25.0 # Position tolerance for successful trajectory (m)
    v_tol = 25.0 # Velocity tolerance for successful trajectory (m/s)
    verbose = True

    results = []
    failed_initial_conditions = []
    np.random.seed(8)
    for i in range(num_runs):
        print(f"--- Starting run {i+1}/{num_runs} ---")

        # Generate random initial conditions for each run
        initial_conditions = np.zeros(7)
        initial_conditions[State.X] = np.random.uniform(-50000.0, 50000.0) # m
        initial_conditions[State.Y] = np.random.uniform(-50000.0, 50000.0) # m
        initial_conditions[State.Z] = planet_params['R'] + np.random.uniform(50000.0, 150000.0) # m
        initial_conditions[State.VX] = np.random.uniform(-1000.0, 1000.0) # m/s
        initial_conditions[State.VY] = np.random.uniform(-1000.0, 1000.0) # m/s
        initial_conditions[State.VZ] = np.random.uniform(-500.0, 500.0) # m/s
        initial_conditions[State.M] = rocket_params['mass_start'] # kg

        # initial_conditions = {'x0': 48045.32429793403, 'y0': 38719.85545747625, 'z0': 6480179.72465158, 'vx0': 951.9382652644483, 'vy0': 968.2924888876025, 'vz0': -271.9402464272812, 'm0': 75000.0}

        print(f"Initial conditions (run {i+1}): {initial_conditions}")

        zem_zev_pdg = ZEMZEVPoweredDescentGuidance(planet_params, rocket_params, None)
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
        comparison = traj_val.compare_reference_vs_simulated_trajectories(zem_zev_pdg.state_dynamics_func, initial_conditions, successful_runs[0], planet_params, rocket_params, N_final)
        traj_val.plot_trajectory_comparison(successful_runs[0], comparison, N_final)
        traj_plot.animate_trajectory(successful_runs[0], N_final, planet_params['R'], thrust_scale=0.005, interval=1, mode='rocket')
    if (num_runs > 1 and num_success > 0):
        traj_plot.plot_multiple_runs(successful_runs)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    main()