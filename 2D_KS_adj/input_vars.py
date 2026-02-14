import numpy as np
from get_vars import get_vars

# define variables 
Lx, Ly = 10, 10                 # domain size
nx, ny = 64, 64                 # number of collocation points
dt = 100                        # only controls what interval we receive the output u_lst and t_lst to be (actual time step is controlled in solve_ivp)

# obtain domain field (x), and fourier wave numbers kx
X, KX, Y, KY = get_vars(2*Lx, 2*Ly, nx, ny)

# define initial conditions of field variable u
u0 = np.cos(np.pi*X/Lx) + np.cos(np.pi*(-X/Lx + 2*Y/Ly)) + np.cos(np.pi*(-X/Lx - 2*Y/Ly))
f = 0
#u0 = np.loadtxt("output_u.csv", delimiter=',')

# define iteration time variables
T1, tol1 = 10, 1e-8
T2, tol2 = 40, 1e-10
T3, tol3 = 1000, 1e-12
T4, tol4 = 10000, 1e-14
T5, tol5 = 20000, 1e-16
stages = ((T1, tol1), (T2, tol2), (T3, tol3), (T4, tol4), (T5, tol5))
stage = 0