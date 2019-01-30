import numpy as np
from numpy import sin, cos

def DoP(I, Q, U, V):
    return np.sqrt(Q**2 + U**2 + V**2) / I

def DoLP(I, Q, U, V):
    return DoP(I, Q, U, V=0.)

def AoLP(I, Q, U, V):
    return 0.5 * np.arctan2(U, Q)

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

def Rotation_matrix_radians(phi):
    return np.array([[1.,          0.,          0., 0.],
                     [0.,  cos(2*phi),  sin(2*phi), 0.],
                     [0., -sin(2*phi),  cos(2*phi), 0.],
                     [0.,          0.,          0., 1.]])

def rotate_element_radians(element, phi):
    return Rotation_matrix_radians(-phi) @ element @ Rotation_matrix_radians(phi)

def Rotation_matrix_degrees(phi_degrees):
    phi_radians = np.deg2rad(phi_degrees)
    return Rotation_matrix_radians(phi_radians)

def rotate_element_degrees(element, phi_degrees):
    phi = np.deg2rad(phi_degrees)
    return rotate_element_radians(element, phi)

Linear_polarizer_0 = 0.5 * np.array([[1, 1, 0, 0],
                                     [1, 1, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])

def Linear_polarizer_radians(phi):
    return rotate_element_radians(Linear_polarizer_0, phi)

def Linear_polarizer_degrees(phi_degrees):
    phi = np.deg2rad(phi_degrees)
    return Linear_polarizer_radians(phi)

def Retarder_radians(delta, phi):
    retarder = np.array([[1, 0,          0,           0],
                         [0, 1,          0,           0],
                         [0, 0, cos(delta), -sin(delta)],
                         [0, 0, sin(delta),  cos(delta)]])
    return rotate_element_radians(retarder, phi)

def Retarder_degrees(delta, phi_degrees):
    phi = np.deg2rad(phi_degrees)
    return Retarder_radians(delta, phi)

def Retarder_frac_deg(d, t):
    d_rad = d * 2 * np.pi
    t_rad = np.deg2rad(t)
    return Retarder_radians(d_rad, t_rad)

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