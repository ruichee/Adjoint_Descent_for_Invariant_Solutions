from get_vars import get_vars

# define variables 
Lx, Ly = 10, 10                 # domain size
nx, ny = 64, 64                 # number of collocation points
dt = 100                        # only controls what interval we receive the output u_lst and t_lst to be (actual time step is controlled in solve_ivp)
f = 0
stage = 0

# obtain domain field (x), and fourier wave numbers kx
X, KX, Y, KY = get_vars(2*Lx, 2*Ly, nx, ny)