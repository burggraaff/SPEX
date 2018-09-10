import numpy as np
from numpy import sin, cos

def DoP(I, Q, U, V):
    return np.sqrt(Q**2 + U**2 + V**2) / I

def DoLP(I, Q, U, V):
    return DoP(I, Q, U, V=0.)

def AoLP(I, Q, U, V):
    return 0.5 * np.arctan(U/Q)

def AoLP_deg(I, Q, U, V):
    return np.rad2deg(AoLP(I, Q, U, V))

def Stokes(I=0., Q=0., U=0., V=0.):
    assert Q**2 + U**2 + V**2 <= I**2
    return np.array([I,Q,U,V], dtype=np.float64)[:,np.newaxis]

def Stokes_from_pol(I0=0, I90=0, I45=0, Im45=0, ILHC=0, IRHC=0):
    return Stokes(I0+I90, I0-I90, I45-Im45, ILHC-IRHC)

def Stokes_nm(I=0., Q=0., U=0., V=0.):
    arr = np.zeros((len(I), 4))
    arr[:,0] = I
    arr[:,1] = Q  # split to allow int/float Q/U/V values
    arr[:,2] = U
    arr[:,3] = V
    return arr

def Rotation(t):
    return np.array([[1.,        0.,        0., 0.],
                     [0.,  cos(2*t), -sin(2*t), 0.],
                     [0.,  sin(2*t),  cos(2*t), 0.],
                     [0.,        0.,        0., 1.]])

def Rotation_deg(t):
    t_rad = np.deg2rad(t)
    return Rotation(t_rad)

LinPol0 = 0.5 * np.array([[1, 1, 0, 0],
                          [1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

def LinPol(angle):
    return Rotation(angle) @ LinPol0 @ Rotation(-angle)

def LinPol_deg(angle):
    angle_rad = np.deg2rad(angle)
    return LinPol(angle_rad)

def Retarder(d, t):
    ret = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, cos(d), sin(d)],
                    [0, 0, -sin(d), cos(d)]])
    return Rotation(t) @ ret @ Rotation(-t)

def Retarder_frac_deg(d, t):
    d_rad = d * 2 * np.pi
    t_rad = np.deg2rad(t)
    return Retarder(d_rad, t_rad)

def Retarder_wavelengths(d_nm, t, wavelengths):
    d_frac = d_nm / wavelengths
    stack = np.stack([Retarder_frac_deg(d, t) for d in d_frac])
    return stack

def Filter(attenuation):
    return attenuation * np.eye(4)

def modulation(wavelengths, source, DoLP, AoLP, delta):
    AoLP_rad = np.deg2rad(AoLP)
    in_cos = 2 * np.pi * delta / wavelengths + 2 * AoLP_rad
    in_brackets = 1 + DoLP * np.cos(in_cos)
    total = 0.5 * source * in_brackets
    return total