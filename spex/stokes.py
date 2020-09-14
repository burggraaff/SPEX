import numpy as np

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
