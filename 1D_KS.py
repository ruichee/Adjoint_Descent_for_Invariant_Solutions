import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def dealiase(f):
    pass


def get_R(u, f, kx):

    # non-linear term u∂ₓu in fourier space
    u_sq = u**2
    u_sqf = np.fft.fft(u_sq**2)
    u_sqf_x = np.complex128(0, kx) * u_sqf
    u_sq_x = np.fft.irfft(u_sqf_x)
    udu = -0.5 * u_sq_x

    # add linear terms in fourier space
    udu_f = np.fft.fft(udu)
    u_f = np.fft.fft(u)
    u_f = dealiase(u_f)
    R = udu_f - (-kx**2 + kx**4)*u_f
    
    return R


def get_G(u, f, kx):
    R = get_R(u, f, kx)
    pass


def adj_descent(u, f, dt, n_iter, tol):

    for _ in range(n_iter):

        un = u.copy()
        G = get_G(u, f)

        u = un + dt*G # can implement rk45 later on
        
        err: float # implement error calc
        if err < tol:
            break

    return u
        

def ngh_descent(u, f, dt, n_iter, tol):
    
    for _ in range(n_iter):

        un = u.copy()
        R = get_R(u, f)
        
        u = un + dt*R # can implement rk45 later on

        err: float # implement error calc
        if err < tol:
            break

    return u


def plot_data():
    pass


def main(u0, L, n, f, dt, n_iter_adj, n_iter_ngh, tol_adj, tol_ngh):

    # given n = EVEN number of collocation points, define grid
    x = np.linspace(0, L, n)
    # fourier wave numbers (k) for DFT
    kx = 2*np.pi/L * np.arange(-n//2, n//2, 1)
    

    u = adj_descent(u0, f, dt, n_iter_adj, tol_adj)
    
    # check if ngh descent is required here

    u = ngh_descent(u, f, dt, n_iter_ngh, tol_ngh)

    plot_data()


