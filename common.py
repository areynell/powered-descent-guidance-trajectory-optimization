from enum import IntEnum

class State(IntEnum):
    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    M = 6

class Control(IntEnum):
    X = 0
    Y = 1
    Z = 2
    GAMMA = 3 # Thrust magnitude slack variable for convex relaxation

class PlanetParams:
    def __init__(self, G, M, R, rho0, H0):
        self.G = G
        self.M = M
        self.R = R
        self.rho0 = rho0
        self.H0 = H0

class RocketParams:
    def __init__(self, mass_total, mass_struct, T_min, T_max, Isp, g0, Aref, Cd):
        self.mass_total = mass_total
        self.mass_struct = mass_struct
        self.T_min = T_min
        self.T_max = T_max
        self.Isp = Isp
        self.g0 = g0
        self.Aref = Aref
        self.Cd = Cd