import numpy as np
from numpy import sin, cos

def DoP(I, Q, U, V):
    return np.sqrt(Q**2 + U**2 + V**2) / I

def DoLP(I, Q, U):
    return DoP(I, Q, U, V=0.)

def AoLP(Q, U):
    return 0.5 * np.arctan(U/Q)

def AoLP_deg(Q, U):
    return np.rad2deg(AoLP(Q, U))

def Stokes(I=0., Q=0., U=0., V=0.):
    return np.array([I,Q,U,V], dtype=np.float64)[:,np.newaxis]

def Stokes_from_pol(I0=0, I90=0, I45=0, Im45=0, ILHC=0, IRHC=0):
    return Stokes(I0+I90, I0-I90, I45-Im45, ILHC-IRHC)

def Rotation(t):
    return np.array([[1.,        0.,       0., 0.],
                     [0.,  cos(2*t), sin(2*t), 0.],
                     [0., -sin(2*t), cos(2*t), 0.],
                     [0.,        0.,       0., 1.]])

def Rotation_deg(t):
    t_rad = np.deg2rad(t)
    return Rotation(t_rad)

LinPol0 = 0.5 * np.array([[1, 1, 0, 0],
                          [1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

def LinPol(angle):
    return Rotation(-angle) @ LinPol0 @ Rotation(angle)

def LinPol_deg(angle):
    angle_rad = np.deg2rad(angle)
    return LinPol(angle_rad)

def Retarder(d, t):
    return np.array([[1., 0., 0., 0.],
                    [0., cos(2*t)**2 + cos(d) * sin(2*t)**2, cos(2*t) * sin(2*t) - cos(2*t) * cos(d) * sin(2*t), sin(2*t) * sin(d)],
                     [0., cos(2*t) * sin(2*t) - cos(2*t) * cos(d) * sin(2*t), cos(d) * cos(2*t)**2 + sin(2*t)**2, -cos(2*t) * sin(d)],
                     [0, -sin(2*t) * sin(d), cos(2*t) * sin(d), cos(d)]])

def Retarder_frac_deg(d, t):
    d_rad = d * 2 * np.pi
    t_rad = np.deg2rad(t)
    return Retarder(d_rad, t_rad)

def Filter(attenuation):
    return attenuation * np.eye(4)