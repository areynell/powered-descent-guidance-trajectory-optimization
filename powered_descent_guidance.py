# TODO:
# 0) Add drag force
# 1) Make 2D animation of trajectory and control input
# 2) Scale variables for faster/more robust convergence
# 3) Use Hermite interpolation polynomial to get smoother trajectory plots
# 4) Plot midpoints as well

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

# System dynamics
def dynamics(state, control, params):
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    m = state[4]

    ux = control[0]
    uy = control[1]

    G = params[0]
    M = params[1]

    # Mass flow rate coefficient
    alpha = 0.001

    # Equations of motion
    dx = vx
    dy = vy
    dvx = -G*M*x/(x**2 + y**2)**(3/2) + ux/m
    dvy = -G*M*y/(x**2 + y**2)**(3/2) + uy/m
    dm = -alpha*ca.sqrt(ux**2 + uy**2 + 1.0e-6)

    return ca.vertcat(dx, dy, dvx, dvy, dm)

def add_hermite_simpson_dynamic_constraints(opti, X, U, N, dt, params):
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

def main():
    # Parameters
    N = 100  # Number of control intervals

    R = 6.371e6    # Radius of Earth (m)
    G = 6.672e-11  # Gravitational constant
    M = 5.972e24   # Mass of Earth (kg)
    params = [G, M]

    # Randomized initial conditions
    t0 = 0.0
    x0 = np.random.uniform(-1000.0, 1000.0) # 100.0
    y0 = R + np.random.uniform(100.0, 1000.0) # R + 100.0
    vx0 = np.random.uniform(-250.0, 250.0) # 100.0
    vy0 = np.random.uniform(-100.0, 0.0) # 0.0
    m0 = 100.0 #  100.0

    # Terminal conditions
    xf = 0.0
    yf = R
    vxf = 0.0
    vyf = 0.0
    mf = 10.0  # Minimum allowable final mass

    # Control bounds
    u_min = -10000.0
    u_max = 10000.0

    # State dimensions
    num_states = 5  # [x, y, vx, vy, m]
    num_controls = 2  # [ux, uy]

    # Optimization variables
    opti = ca.Opti()

    # Time horizon (free)
    tf = opti.variable()
    opti.subject_to(tf > 0)

    # Time step
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
    opti.subject_to(x[-1] == xf)
    opti.subject_to(y[-1] == yf)
    opti.subject_to(vx[-1] == vxf)
    opti.subject_to(vy[-1] == vyf)
    opti.subject_to(m[-1] >= mf)

    # Control constraints
    opti.subject_to(opti.bounded(u_min, ux, u_max))
    opti.subject_to(opti.bounded(u_min, uy, u_max))

    # Mass constraints (mass cannot increase)
    for k in range(N):
        opti.subject_to(m[k+1] <= m[k])

    # add_hermite_simpson_dynamic_constraints(opti, X, U, N, dt, params)
    add_gauss_lobatto_dynamic_constraints(opti, X, U, N, dt, params)

    # Objective: Maximize final mass (by minimizing negative final mass)
    opti.minimize(-m[-1])

    # Initial guess for decision variables
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
            opti.set_initial(ux[i], 0.0)
            opti.set_initial(uy[i], 0.0)

    # Set up solver
    opti.solver('ipopt', {
        'ipopt.print_level': 5,
        'expand': True,
        'ipopt.max_iter': 5000,
        'ipopt.tol': 1e-6
    })

    # Solve the optimization problem
    print("Solving optimization problem...")
    t_start = time()
    sol = opti.solve()
    t_end = time()
    print(f"Solution found in {t_end - t_start:.2f} seconds")

    # Extract solution
    tf_opt = sol.value(tf)
    x_opt = sol.value(x)
    y_opt = sol.value(y)
    vx_opt = sol.value(vx)
    vy_opt = sol.value(vy)
    m_opt = sol.value(m)
    ux_opt = sol.value(ux)
    uy_opt = sol.value(uy)

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
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Control [N]')
    plt.legend()

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

    plt.show()

if __name__ == "__main__":
    main()