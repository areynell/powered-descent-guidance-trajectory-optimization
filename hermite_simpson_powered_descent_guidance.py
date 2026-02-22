# TODO:
# 0) Plot planet's surface near landing site, and add linear surface constraint instead of radial constraint
# 1) Split path constraints, initial guesses, and objective function into separate functions for clarity
# 2) Add glide-slope constraint near landing site
# 3) Make 2D animation of trajectory and control input
# 4) Use polynomial trajectories for state and controls that match collocation method
# 5) Plot midpoints as well
# 6) Try different collocation methods
# 7) Add rotation of planet
# 8) Re-order functions to make reading flow easier
# 9) Add 6-DOF dynamics

import numpy as np
import casadi as ca
from time import time

import trajectory_validation as traj_val
import trajectory_plotting as traj_plot
from zem_zev_powered_descent_guidance import ZEMZEVPoweredDescentGuidance

from common import State, Control

def dynamics_scaled(state_scaled, control_scaled, planet_params, rocket_params, bounds, scale_factors, offsets):
    """
    Scaled dynamics in non-dimensional units.
    state = [x, y, z, vx, vy, vz, m] (scaled)
    control = [ux, uy, uz, gamma] (scaled)
    returns scaled derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dm/dt]
    computed by unscaling the inputs, calling dynamics(), and rescaling the outputs.
    """
    T = scale_factors['t']
    S_x = np.vstack((scale_factors['x'],
                     scale_factors['y'],
                     scale_factors['z'],
                     scale_factors['vx'],
                     scale_factors['vy'],
                     scale_factors['vz'],
                     scale_factors['m']))
    S_u = np.vstack((scale_factors['ux'],
                     scale_factors['uy'],
                     scale_factors['uz'],
                     scale_factors['gamma']))

    x_offsets = np.vstack((offsets['x'],
                           offsets['y'],
                           offsets['z'],
                           offsets['vx'],
                           offsets['vy'],
                           offsets['vz'],
                           offsets['m']))
    u_offsets = np.vstack((offsets['ux'],
                           offsets['uy'],
                           offsets['uz'],
                           offsets['gamma']))

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
    x = state[State.X]
    y = state[State.Y]
    z = state[State.Z]
    vx = state[State.VX]
    vy = state[State.VY]
    vz = state[State.VZ]
    m = state[State.M]

    ux = control[Control.X]
    uy = control[Control.Y]
    uz = control[Control.Z]
    gamma = control[Control.GAMMA]

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
    eps = 1.0e-8
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

class HermiteSimpsonPoweredDescentGuidance:
    def __init__(self, planet_params: dict, rocket_params: dict, state_dynamics_func: callable):
        self.planet_params = planet_params
        self.rocket_params = rocket_params
        self.state_dynamics_func = state_dynamics_func

        self.opti = None
        self.opt_vars_scaled = None

        self.scale_factors = None
        self.offsets = None
        self.initial_conditions = None
        self.initial_conditions_scaled = None
        self.terminal_conditions = None
        self.terminal_conditions_scaled = None
        self.landing_tolerances = None
        self.landing_tolerances_scaled = None
        self.bounds = None
        self.bounds_scaled = None
        self.N = None

    def generate_trajectory(self, initial_conditions: np.ndarray, terminal_conditions: np.ndarray, bounds: dict, landing_tolerances: dict, scale_factors: dict, offsets: dict, N: int):
        """
        Main entry point to solve the trajectory.
        """

        self.opti = ca.Opti()
        self.initial_conditions = initial_conditions
        self.terminal_conditions = terminal_conditions
        self.bounds = bounds
        self.scale_factors = scale_factors
        self.offsets = offsets
        self.N = N

        # Creating scaled variables to improve convergence
        self.opt_vars_scaled = self._create_scaled_optimization_variables()
        self.initial_conditions_scaled = self._scale_state(initial_conditions, scale_factors, offsets)
        self.terminal_conditions_scaled = self._scale_state(terminal_conditions, scale_factors, offsets)
        self.bounds_scaled = self._scale_bounds(bounds, scale_factors, offsets)
        self.landing_tolerances_scaled = self._scale_landing_tolerances(landing_tolerances, scale_factors)

        self._add_boundary_constraints()
        self._add_path_constraints()
        self._add_hermite_simpson_dynamic_constraints()
        self._add_objective_function()
        self._compute_initial_solution_guess()

        return self._solve()

    def _create_scaled_optimization_variables(self):
        # Scaled state variables = [x, y, z, vx, vy, vz, m]
        self.num_states = 7

        # Scaled control variables = [ux, uy, uz, gamma]
        # NOTE: Gamma is a thrust magnitude slack variable for convex relaxation
        self.num_controls = 4

        return {
            'X': self.opti.variable(self.num_states, self.N+1),
            'U': self.opti.variable(self.num_controls, self.N+1),
            'tf': self.opti.variable(),
            'slack_surface': self.opti.variable(1, self.N+1)
        }

    def _scale_state(self, state, scale_factors: dict, offsets: dict):
        x = (state[State.X] - offsets['x']) / scale_factors['x']
        y = (state[State.Y] - offsets['y']) / scale_factors['y']
        z = (state[State.Z] - offsets['z']) / scale_factors['z']
        vx = (state[State.VX] - offsets['vx']) / scale_factors['vx']
        vy = (state[State.VY] - offsets['vy']) / scale_factors['vy']
        vz = (state[State.VZ] - offsets['vz']) / scale_factors['vz']
        m = (state[State.M] - offsets['m']) / scale_factors['m']

        return np.array([x, y, z, vx, vy, vz, m])

    def _scale_states(self, state, scale_factors: dict, offsets: dict):
        x =(state[State.X, :] - offsets['x']) / scale_factors['x']
        y = (state[State.Y, :] - offsets['y']) / scale_factors['y']
        z = (state[State.Z, :] - offsets['z']) / scale_factors['z']
        vx = (state[State.VX, :] - offsets['vx']) / scale_factors['vx']
        vy = (state[State.VY, :] - offsets['vy']) / scale_factors['vy']
        vz = (state[State.VZ, :] - offsets['vz']) / scale_factors['vz']
        m = (state[State.M, :] - offsets['m']) / scale_factors['m']

        if isinstance(state, (ca.MX, ca.SX, ca.DM)):
            print("CasADi state variable detected, using vertcat for scaling")
            return ca.vertcat(x, y, z, vx, vy, vz, m)
        else:
            print("NumPy state variable detected, using vstack for scaling")
            return np.vstack([x, y, z, vx, vy, vz, m])

    def _unscale_states(self, state_scaled, scale_factors: dict, offsets: dict):
        x = state_scaled[State.X, :] * scale_factors['x'] + offsets['x']
        y = state_scaled[State.Y, :] * scale_factors['y'] + offsets['y']
        z = state_scaled[State.Z, :] * scale_factors['z'] + offsets['z']
        vx = state_scaled[State.VX, :] * scale_factors['vx'] + offsets['vx']
        vy = state_scaled[State.VY, :] * scale_factors['vy'] + offsets['vy']
        vz = state_scaled[State.VZ, :] * scale_factors['vz'] + offsets['vz']
        m = state_scaled[State.M, :] * scale_factors['m'] + offsets['m']

        if isinstance(state_scaled, (ca.MX, ca.SX, ca.DM)):
            print("CasADi state variable detected, using vertcat for unscaling")
            return ca.vertcat(x, y, z, vx, vy, vz, m)
        else:
            print("NumPy state variable detected, using vstack for unscaling")
            return np.vstack([x, y, z, vx, vy, vz, m])

    def _scale_control(self, control, scale_factors: dict, offsets: dict):
        ux = (control[Control.X] - offsets['ux']) / scale_factors['ux']
        uy = (control[Control.Y] - offsets['uy']) / scale_factors['uy']
        uz = (control[Control.Z] - offsets['uz']) / scale_factors['uz']
        gamma = (control[Control.GAMMA] - offsets['gamma']) / scale_factors['gamma']

        return np.array([ux, uy, uz, gamma])

    def _scale_controls(self, control, scale_factors: dict, offsets: dict):
        ux = (control[Control.X, :] - offsets['ux']) / scale_factors['ux']
        uy = (control[Control.Y, :] - offsets['uy']) / scale_factors['uy']
        uz = (control[Control.Z, :] - offsets['uz']) / scale_factors['uz']
        gamma = (control[Control.GAMMA, :] - offsets['gamma']) / scale_factors['gamma']

        if isinstance(control, (ca.MX, ca.SX, ca.DM)):
            print("CasADi control variable detected, using vertcat for scaling")
            return ca.vertcat(ux, uy, uz, gamma)
        else:
            print("NumPy control variable detected, using vstack for scaling")
            return np.vstack([ux, uy, uz, gamma])

    def _unscale_controls(self, control_scaled, scale_factors: dict, offsets: dict):
        ux = control_scaled[Control.X, :] * scale_factors['ux'] + offsets['ux']
        uy = control_scaled[Control.Y, :] * scale_factors['uy'] + offsets['uy']
        uz = control_scaled[Control.Z, :] * scale_factors['uz'] + offsets['uz']
        gamma = control_scaled[Control.GAMMA, :] * scale_factors['gamma'] + offsets['gamma']

        if isinstance(control_scaled, (ca.MX, ca.SX, ca.DM)):
            print("CasADi control variable detected, using vertcat for unscaling")
            return ca.vertcat(ux, uy, uz, gamma)
        else:
            print("NumPy control variable detected, using vstack for unscaling")
            return np.vstack([ux, uy, uz, gamma])

    def _scale_time(self, time: float, scale_factors: dict, offsets: dict):
        return (time - offsets['t']) / scale_factors['t']

    def _unscale_time(self, time_scaled: float, scale_factors: dict, offsets: dict):
        return time_scaled * scale_factors['t'] + offsets['t']

    def _scale_landing_tolerances(self, landing_tolerances: dict, scale_factors: dict):
        return {
            'position': (landing_tolerances['position']) / scale_factors['x'],
            'velocity': (landing_tolerances['velocity']) / scale_factors['vx'],
        }

    def _scale_bounds(self, bounds: dict, scale_factors: dict, offsets: dict):
        return {
            'u_min': (bounds['u_min'] - offsets['gamma']) / scale_factors['gamma'],
            'u_max': (bounds['u_max'] - offsets['gamma']) / scale_factors['gamma'],
            't_min': (bounds['t_min'] - offsets['t']) / scale_factors['t'],
            't_max': (bounds['t_max'] - offsets['t']) / scale_factors['t']
        }

    def _add_boundary_constraints(self):
        # State constraints at initial and final time steps
        X_scaled = self.opt_vars_scaled['X']
        assert(X_scaled.shape == (self.num_states, self.N+1))

        # Initial state constraints
        X0_scaled = self.initial_conditions_scaled
        self.opti.subject_to(X_scaled[:, 0] == X0_scaled)

        # Final state constraints
        Xf_scaled = self.terminal_conditions_scaled
        pos_tol = self.landing_tolerances_scaled['position']
        vel_tol = self.landing_tolerances_scaled['velocity']
        self.opti.subject_to(self.opti.bounded(Xf_scaled[State.X:State.Z+1] - pos_tol, X_scaled[State.X:State.Z+1, -1], Xf_scaled[State.X:State.Z+1] + pos_tol))
        self.opti.subject_to(self.opti.bounded(Xf_scaled[State.VX:State.VZ+1] - vel_tol, X_scaled[State.VX:State.VZ+1, -1], Xf_scaled[State.VX:State.VZ+1] + vel_tol))

        # self.opti.subject_to(self.opti.bounded(Xf_scaled[State.X] - pos_tol, X_scaled[State.X, -1], Xf_scaled[State.X] + pos_tol))
        # self.opti.subject_to(self.opti.bounded(Xf_scaled[State.Y] - pos_tol, X_scaled[State.Y, -1], Xf_scaled[State.Y] + pos_tol))
        # self.opti.subject_to(self.opti.bounded(Xf_scaled[State.Z] - pos_tol, X_scaled[State.Z, -1], Xf_scaled[State.Z] + pos_tol))
        # self.opti.subject_to(self.opti.bounded(Xf_scaled[State.VX] - vel_tol, X_scaled[State.VX, -1], Xf_scaled[State.VX] + vel_tol))
        # self.opti.subject_to(self.opti.bounded(Xf_scaled[State.VY] - vel_tol, X_scaled[State.VY, -1], Xf_scaled[State.VY] + vel_tol))
        # self.opti.subject_to(self.opti.bounded(Xf_scaled[State.VZ] - vel_tol, X_scaled[State.VZ, -1], Xf_scaled[State.VZ] + vel_tol))

        # Time constraint
        tf_scaled = self.opt_vars_scaled['tf']
        self.opti.subject_to(self.opti.bounded(self.bounds_scaled['t_min'], tf_scaled, self.bounds_scaled['t_max']))

    def _add_path_constraints(self):
        # Keep vehicle above surface (r >= R)
        X_scaled = self.opt_vars_scaled['X']
        # Using physical units since x, y and z scales might differ
        X = self._unscale_states(X_scaled, self.scale_factors, self.offsets)
        slack_surface = self.opt_vars_scaled['slack_surface']
        self.opti.subject_to(slack_surface >= 0.0)
        # IMPORTANT: Normalizing by R to keep gradients well-scaled (r >= R => r/R >= 1), and using slack variable to improve convergence
        self.opti.subject_to(ca.sqrt(X[State.X]**2 + X[State.Y]**2 + X[State.Z]**2) / self.planet_params['R'] >= 1.0 - slack_surface)

        # Control constraints
        U_scaled = self.opt_vars_scaled['U']
        self.opti.subject_to(self.opti.bounded(-self.bounds_scaled['u_max'], U_scaled[Control.X, :], self.bounds_scaled['u_max']))
        self.opti.subject_to(self.opti.bounded(-self.bounds_scaled['u_max'], U_scaled[Control.Y, :], self.bounds_scaled['u_max']))
        self.opti.subject_to(self.opti.bounded(-self.bounds_scaled['u_max'], U_scaled[Control.Z, :], self.bounds_scaled['u_max']))
        self.opti.subject_to(self.opti.bounded(self.bounds_scaled['u_min'], U_scaled[Control.GAMMA, :], self.bounds_scaled['u_max']))
        # Convex relaxation of thrust magnitude
        self.opti.subject_to(U_scaled[Control.X, :]**2 + U_scaled[Control.Y, :]**2 + U_scaled[Control.Z, :]**2 <= U_scaled[Control.GAMMA, :]**2)

        # Mass constraints
        m_scaled = X_scaled[State.M, :]
        mf_scaled = self.terminal_conditions_scaled[State.M]
        # Keep mass above dry-mass lower bound everywhere
        self.opti.subject_to(m_scaled >= mf_scaled)

        # NOTE: This constraint really seems to help produce faster convergence and avoid infeasible solutions, why?
        # Mass cannot increase over time
        for k in range(self.N):
            self.opti.subject_to(m_scaled[k+1] <= m_scaled[k])

    def _add_hermite_simpson_dynamic_constraints(self):
        """
        Add dynamic constraints using Hermite-Simpson collocation.
        """

        # Time step
        # NOTE: Is there a better way we can incorporate a free final time? "tdot = 1" or "time scaling"?
        tf_scaled = self.opt_vars_scaled['tf']
        dt_scaled = tf_scaled / self.N

        for k in range(self.N):
            xk = self.opt_vars_scaled['X'][:, k]
            uk = self.opt_vars_scaled['U'][:, k]
            xk_next = self.opt_vars_scaled['X'][:, k+1]
            uk_next = self.opt_vars_scaled['U'][:, k+1]

            # Dynamics at beginning and end of interval
            fk = dynamics_scaled(xk, uk, self.planet_params, self.rocket_params, self.bounds, self.scale_factors, self.offsets)
            fk_next = dynamics_scaled(xk_next, uk_next, self.planet_params, self.rocket_params, self.bounds, self.scale_factors, self.offsets)

            # Midpoint state (Hermite interpolation)
            xk_mid = 0.5 * (xk + xk_next) + (dt_scaled/8.0) * (fk - fk_next)

            # Midpoint control (linear interpolation)
            uk_mid = 0.5 * (uk + uk_next)

            # Dynamics at midpoint
            fk_mid = dynamics_scaled(xk_mid, uk_mid, self.planet_params, self.rocket_params, self.bounds, self.scale_factors, self.offsets)

            # Simpson integration
            self.opti.subject_to(xk_next == xk + (dt_scaled/6.0) * (fk + 4.0*fk_mid + fk_next))

    def _compute_initial_solution_guess(self):
        X_scaled = self.opt_vars_scaled['X']
        U_scaled = self.opt_vars_scaled['U']
        tf_scaled = self.opt_vars_scaled['tf']
        slack_surface = self.opt_vars_scaled['slack_surface']

        # Using ZEM/ZEV guidance law to generate an initial guess trajectory for the constrained trajectory optimization
        zem_zev_pdg = ZEMZEVPoweredDescentGuidance(self.planet_params, self.rocket_params, dynamics_np)
        k = 1 # Factor to increase number of integration steps for final trajectory once optimal time of flight is found in line search
        N_final = k*self.N # Number of integration steps for final trajectory validation and plotting
        N_pred = 10 # Number of integration steps for ZEM/ZEV prediction
        t_min = self.bounds['t_min'] # Minimum time of flight in line search (s)
        t_max = self.bounds['t_max'] # Maximum time of flight in line search (s)
        t_tol = 1.0 # Optimal time of flight tolerance in line search (s)
        r_tol = 100.0 # Only require roughly correct terminal position for initial guess, since optimization will refine it
        v_tol = 100.0 # Only require roughly correct terminal velocity for initial guess, since optimization will refine it
        zem_zev_trajectory = zem_zev_pdg.find_best_trajectory(self.initial_conditions, self.terminal_conditions, t_min, t_max, t_tol, r_tol, v_tol, self.N, N_final, N_pred)

        if (zem_zev_trajectory['success']):
            print("Valid ZEM/ZEV trajectory found with time of flight:", zem_zev_trajectory['tf'])

            stride = k # Sub-sample the ZEM/ZEV trajectory to match the number of optimization nodes (N+1)

            tf_guess = zem_zev_trajectory['tf']
            X_guess = zem_zev_trajectory['X'][::stride]

            U_guess = zem_zev_trajectory['U'][::stride]
            U_guess_gamma = np.sqrt(U_guess[Control.X]**2 + U_guess[Control.Y]**2 + U_guess[Control.Z]**2)
            U_guess = np.vstack((U_guess, U_guess_gamma)) # Add gamma as the last row of the control guess for convex relaxation

            tf_guess_scaled = self._scale_time(tf_guess, self.scale_factors, self.offsets)
            X_guess_scaled = self._scale_states(X_guess, self.scale_factors, self.offsets)
            U_guess_scaled = self._scale_controls(U_guess, self.scale_factors, self.offsets)

            # Initial guess solution: Use ZEM/ZEV trajectory for states, controls and final time
            self.opti.set_initial(tf_scaled, tf_guess_scaled)
            self.opti.set_initial(X_scaled, X_guess_scaled)
            self.opti.set_initial(U_scaled, U_guess_scaled)
        else:
            print("Valid ZEM/ZEV trajectory not found, using linear interpolation initial guess instead")
            # Initial guess solution: Linear interpolation between boundary conditions for states, and min thrust for controls
            X0_scaled = self.initial_conditions_scaled
            Xf_scaled = self.terminal_conditions_scaled

            # Initial guess for time horizon
            self.opti.set_initial(tf_scaled, (self.bounds_scaled['t_min'] + self.bounds_scaled['t_max']) / 2.0)
            for i in range(self.N+1):
                t_frac = i / self.N
                # Linear interpolation initial guess for states
                self.opti.set_initial(X_scaled[:, i], (1.0 - t_frac)*X0_scaled + t_frac*Xf_scaled)
                # Min thrust initial guess for controls
                self.opti.set_initial(U_scaled[:, i], self.bounds_scaled['u_min'])

        # Initialize slack on surface constraint to zero
        self.opti.set_initial(slack_surface, 0.0)

    def _add_objective_function(self):
        m_scaled = self.opt_vars_scaled['X'][State.M, :]
        slack_surface = self.opt_vars_scaled['slack_surface']

        # Objective function: Maximize final mass (by minimizing fuel use) AND minimize slack
        # NOTE: Using slack variables on position and velocity seemed to worsen convergence. Better to use tolerances on terminal constraints.
        surface_slack_penalty = 1.0e6
        self.opti.minimize(-m_scaled[-1] + surface_slack_penalty * ca.sumsqr(slack_surface))

    def _solve(self):
        # Set up solver
        ipopt_options = {
            'expand': True,
            'ipopt.print_level': 5,
            'ipopt.max_iter': 500,
            'ipopt.tol': 1e-6,
            'ipopt.linear_solver': 'mumps',
            'ipopt.hessian_approximation': 'exact',
            'ipopt.nlp_scaling_method': 'gradient-based',
            # 'ipopt.nlp_scaling_method': 'none',
            # 'ipopt.mu_strategy': 'adaptive',
        }
        self.opti.solver('ipopt', ipopt_options)

        try:
            # Solve the optimization problem
            t_start = time()
            sol = self.opti.solve()
            t_end = time()
            t_solve = t_end - t_start

            solution_stats = self.opti.stats()
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

            X_scaled = self.opt_vars_scaled['X']
            U_scaled = self.opt_vars_scaled['U']
            tf_scaled = self.opt_vars_scaled['tf']

            print("Min scaled state magnitudes:", np.min(sol.value(X_scaled), axis=1))
            print("Max scaled state magnitudes:", np.max(sol.value(X_scaled), axis=1))
            print("Min scaled control magnitudes:", np.min(sol.value(U_scaled), axis=1))
            print("Max scaled control magnitudes:", np.max(sol.value(U_scaled), axis=1))
            print("Final scaled time:", sol.value(tf_scaled))

            # Extract and unscale solution
            X = self._unscale_states(sol.value(X_scaled), self.scale_factors, self.offsets)
            U = self._unscale_controls(sol.value(U_scaled), self.scale_factors, self.offsets)
            tf = self._unscale_time(sol.value(tf_scaled), self.scale_factors, self.offsets)

            print(f"--- Run succeeded ---")
            print(f"Converged in {iterations} iterations ({t_solve:.2f} seconds)")
            print(f"Final time: {tf:.2f} s")
            print(f"Final mass: {X[State.M, -1]:.2f} kg")
            return {
                "success": True,
                "iterations": iterations,
                "solve_time": t_solve,
                "tf": tf,
                "X": X,
                "U": U,
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
    terminal_conditions_dict = {
        'x': 0.0, # Final x-position (m)
        'y': 0.0, # Final y-position (m)
        'z': planet_params['R'], # Final z-position (m)
        'vx': 0.0, # Final x-velocity (m/s)
        'vy': 0.0, # Final y-velocity (m/s)
        'vz': 0.0, # Final z-velocity (m/s)
        'm': rocket_params['mass_dry'] # Minimum ALLOWABLE final mass (kg)
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
        'position': 1.0, # (m)
        'velocity': 0.5, # (m/s)
    }

    N = 50 # Number of control intervals
    num_runs = 100 # Number of randomized runs to perform

    results = []
    failed_initial_conditions = []
    t_start_total = time()
    np.random.seed(8)
    for i in range(num_runs):
        print(f"--- Starting run {i+1}/{num_runs} ---")
        # Randomized initial conditions
        initial_conditions_dict = {
            'x': np.random.uniform(-50000.0, 50000.0), # m
            'y': np.random.uniform(-50000.0, 50000.0), # m
            'z': planet_params['R'] + np.random.uniform(50000.0, 150000.0), # m
            'vx': np.random.uniform(-1000.0, 1000.0), # m/s
            'vy': np.random.uniform(-1000.0, 1000.0), # m/s
            'vz': np.random.uniform(-500.0, 500.0), # m/s
            'm': rocket_params['mass_start'] # kg - Landing mass (Dry mass + reserve)
        }

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

        print(f"Initial conditions (run {i+1}): {initial_conditions_dict}")

        hermite_simpson_pdg = HermiteSimpsonPoweredDescentGuidance(planet_params, rocket_params, dynamics_scaled)
        initial_conditions = np.array(list(initial_conditions_dict.values()))
        terminal_conditions = np.array(list(terminal_conditions_dict.values()))
        result = hermite_simpson_pdg.generate_trajectory(initial_conditions, terminal_conditions, bounds, landing_tolerances, scale_factors, offsets, N)
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
        # comparison = traj_val.compare_optimized_vs_simulated_trajectories(dynamics_np, initial_conditions_dict, successful_runs[0], planet_params, rocket_params, N)
        # traj_val.plot_trajectory_comparison(successful_runs[0], comparison, N)
        traj_plot.animate_trajectory(successful_runs[0], N, planet_params['R'], thrust_scale=0.005, interval=1)
    elif (num_runs > 1 and num_success > 0):
        traj_plot.plot_multiple_runs(successful_runs)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    main()