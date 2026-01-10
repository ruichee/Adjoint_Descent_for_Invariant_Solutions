import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def dealiase(ff, kx):
    k = np.absolute(kx)
    k_max = 1/3 * np.max(k)
    ff_filtered = np.where(k <= k_max, ff, 0)
    return ff_filtered


def get_R(u, f, kx):

    # non-linear term u∂ₓu in fourier space
    u_sq = u**2
    u_sqf = np.fft.fft(u_sq**2)
    u_sqf_x = -1j * kx * u_sqf
    u_sq_x = np.fft.ifft(u_sqf_x)
    udu = -0.5 * u_sq_x

    # add linear terms in fourier space
    udu_f = np.fft.fft(udu)
    u_f = np.fft.fft(u)
    u_f = dealiase(u_f, kx)
    R_f = udu_f - (-kx**2 + kx**4)*u_f
    R_f = dealiase(R_f, kx)
    
    # set mean flow = 0, i.e. wave has no DC component (constant offset)
    R_f = np.where(kx == 0, 0, R_f)

    # convert back to physical space
    R = np.real(np.fft.ifft(R_f))
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


L = 22
n = 128
x = np.linspace(0, L, n)
m=2
u0=2*np.sin(m*2*np.pi*x/L)
kx = 2*np.pi * np.fft.fftfreq(n, d=L/n)

u = get_R(u0, 0, kx)
plt.plot(u)
plt.show()