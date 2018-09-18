import numpy as np

def D_err(D, D_real):
    return D/D_real - 1

def A_err(A, A_real):
    diff1 = np.abs(A - A_real)
    diff2 = np.abs(A - A_real + 180)
    diff3 = np.abs(A - A_real - 180)
    diff = np.stack((diff1, diff2, diff3)).min(axis=0)
    return diff

def margin(x, DoLPs, AoLPs, real_DoLP, real_AoLP, Dlim=0.03, Alim=5):
    D = D_err(DoLPs, real_DoLP)
    A = A_err(AoLPs, real_AoLP)

    D_ind = np.where(np.abs(D) > Dlim)
    A_ind = np.where(np.abs(A) > Alim)

    try:
        D_min = np.abs(x[D_ind]).min()
    except ValueError:
        D_min = np.abs(x).max()
    try:
        A_min = np.abs(x[A_ind]).min()
    except ValueError:
        A_min = np.abs(x).max()

    x_min = np.min([D_min, A_min])

    return x_min