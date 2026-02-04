import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import axes3d

###############################################################################################

def get_vars(Lx, Ly, nx, ny):

    dx = Lx/nx                                  # define x spatial step
    dy = Ly/ny                                  # define x spatial step
    
    x = np.linspace(0, Lx-dx, nx)               # nx = EVEN no. of collocation points, define grid
    y = np.linspace(0, Ly-dy, ny)               # ny = EVEN no. of collocation points, define grid
    
    kx = 2*np.pi * np.fft.fftfreq(nx, d=Lx/nx)  # fourier wave numbers (kx) for DFT in x-dir
    ky = 2*np.pi * np.fft.fftfreq(ny, d=Ly/ny)  # fourier wave numbers (ky) for DFT in y-dir
    
    KX, KY = np.meshgrid(kx, ky)                # meshgrid of all combinations of kx and ky waves
    X, Y = np.meshgrid(x, y)                    # meshgrid of all combinations of x and y values
    
    return (X, KX, Y, KY)                       # NOTE: L-dx ensure no cutting into next period

###############################################################################################

def dealiase(ff):
    
    global KX, KY

    kx_abs = np.absolute(KX)
    ky_abs = np.absolute(KY)

    kx_max = 2/3 * np.max(kx_abs)                       # maximum frequency that we will keep
    ky_max = 2/3 * np.max(ky_abs)                       # maximum frequency that we will keep

    ff_filterx = np.where(np.abs(KX) < kx_max, ff, 0)           # all higher frequencies in x are set to 0
    ff_filterxy = np.where(np.abs(KY) < ky_max, ff_filterx, 0)  # all higher frequencies in y are set to 0
    
    return ff_filterxy

###############################################################################################

def get_R(t, u): 

    global KX, KY, f

    # obtain u in fourier space
    u_f = np.fft.fft2(u)                        # bring u into fourier

    # non-linear term -1/2(∂ₓu)^2 in fourier space 
    u_x_f = 1j * KX * u_f                       # ∂ₓu in fourier, differentiate via multiply ik_x
    u_y_f = 1j * KY * u_f                       # ∂ᵧu in fourier, differentiate via multiply ik_y
    u_x = np.real(np.fft.ifft2(u_x_f))          # bring back to physical space
    u_y = np.real(np.fft.ifft2(u_y_f))          # bring back to physical space
    u_sq_terms = -0.5 * (u_x*u_x + u_y*u_y)     # get -1/2(∂ₓu)^2

    # linear terms -∂ₓₓu-∂ᵧᵧu-∂ₓₓₓₓu-∂ᵧᵧᵧᵧu-2∂ₓₓ∂ᵧᵧu in fourier space 
    lin_terms_f =  (KX**2 + KY**2 
                    - KX**4 - KY**4 
                    - 2 * KX**2 * KY**2)*u_f    # n-derivative = multiply u by (ik)^n
    
    # add terms together 
    R_f = np.fft.fft2(u_sq_terms) + lin_terms_f
    R_f = dealiase(R_f)                         # dealise R

    # set mean flow = 0, no DC component/offset
    mask = (KX==0) * (KY==0)
    R_f = np.where(mask, 0, R_f)               # ensures the sine wave has no constant component (kx=0 and ky=0)

    # convert back to physical space
    R = np.real(np.fft.ifft2(R_f)) + f         # obtain R(u)

    print(t)
    
    return R

###############################################################################################

def steady_state_event(t, u):

    # rhs can either be get_R() or get_G()
    dudt = get_R(t, u)

    # compute R or G as a magnitude 
    change_in_u = np.linalg.norm(dudt)

    # set tolerance for ending iteration
    global u_tol
    tolerance = u_tol

    # compare, if G or R < tol, end iteration
    return change_in_u - tolerance

# Configure the Event
steady_state_event.terminal = True  # Stop the simulation when this event occurs
steady_state_event.direction = -1   # Only trigger when going from positive -> negative

###############################################################################################

def time_marching(u0, rtol, atol):

    global f, T, dt, nx, ny

    # Set up the time interval
    nt = int(T / dt) + 1  
    tspan = np.linspace(0, T, nt)

    # Integration: use solve_ivp with method='BDF' to mimic ode15s (stiff solver)
    solution = solve_ivp(
        fun=lambda t,u: \
            get_R(t, u.reshape(nx, ny))
            .flatten(),                     # function that returns du/dt
        t_span=(0, T),                      # (start_time, end_time)
        y0=u0.flatten(),                    # Initial condition
        method='Radau',                       # 'BDF' or 'Radau' - implicit adaptive time stepping
        #events=steady_state_event,          # check if ||R(u)|| < tol, can end iteration early
        t_eval=tspan,                       # The specific time points returned
        rtol=rtol,                          # Relative tolerance
        atol=atol                           # Absolute tolerance
    )

    # Extract the output list of iteration values
    u_lst = np.array([u.reshape(nx, ny) for u in solution.y.T])
    t_lst = solution.t.T

    # check for convergence
    return u_lst, t_lst

###############################################################################################

def compute_residuals(t_lst, u_lst):

    G_lst = np.zeros(len(u_lst))

    for i in range(len(u_lst)-1):
        G_lst[i] = np.linalg.norm(get_R(t_lst[i], u_lst[i]))

    return G_lst

###############################################################################################
'''
def plot_data(u_lst, t_lst) -> None:
    
    def update():
        pass
    
    # animate convergence
    #ani = FuncAnimation(fig=fig, frames=update)

    fig, (u_val, res) = plt.subplots(1, 2, figsize=(10, 5))

    #[u_val.plot(u_lst[i]) for i in range(1, len(u_lst))]
    global x
    u_val.plot(x, u_lst[-1])
    u_val.plot(x, u_lst[0], linestyle='--', color='red')
    u_val.set_xlabel('x')
    u_val.set_ylabel('u')
    u_val.set_title('Steady Solution of 1D KS Equation')
    u_val.set_xlim(0, L)
    u_val.grid()
    
    G_lst = compute_residuals(t_lst, u_lst)
    res.plot(t_lst, G_lst)
    res.semilogy()
    res.set_xlabel('τ')
    res.set_title('Residual of Adjoint Norm ||G(u)||')
    res.set_xlim(0, t_lst[-1])
    res.grid()

    fig.tight_layout()
    plt.show()
'''
###############################################################################################

def main(u0, adj_rtol, adj_atol):

    u_lst, t_lst = time_marching(u0, adj_rtol, adj_atol)

    # plot final contour and residual convergence
    fig, (u_val, res) = plt.subplots(1, 2, figsize=(10, 5))
    u_val.contourf(X, Y, u_lst[-1])
    G_lst = compute_residuals(t_lst, u_lst)
    res.plot(t_lst, G_lst)
    res.semilogy()
    res.set_xlabel('τ')
    res.set_title('Residual of L2-Norm ||R(u)||')
    res.set_xlim(0, t_lst[-1])
    res.set_ylim(1e-15, 1e3)
    res.grid()
    plt.show()

    # plot 3D surface and contour
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    Z = u_lst[-1]
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_proj_type('ortho')
    ax.contour(X, Y, Z, levels=8, lw=3, linestyles="solid", offset=-4)
    ax.set_xlim(0, 2*Lx)
    ax.set_ylim(0, 2*Ly)
    ax.set_zlim(-6, 6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("v(x,y)")
    plt.show()

    # check fourier values
    u_k = np.fft.fft2(u_lst[-1])
    func = lambda x,y: np.round(np.abs(u_k[x,y]), 2)
    print(func(0, 1), func(1, 1), func(1, 0))
    print(func(2, 0), func(2, 1), func(3, 0), 
          func(3, 1), func(0, 2), func(1, 2), func(2, 2))
    
    return u_lst, t_lst

###############################################################################################

# define variables 
epsilon = 0.1
v1, v2 = 1-epsilon, 1-epsilon

Lx, Ly = np.pi / np.sqrt(v1), np.pi / np.sqrt(v2)
nx, ny = 64, 64                 # number of collocation points
T = 25                         # max iteration time
dt = 0.05                         # iteration step 
u_tol = 1e-10                    # tolerance for converged u

# obtain domain field (x), and fourier wave numbers kx
X, KX, Y, KY = get_vars(2*Lx, 2*Ly, nx, ny)

# define forcing actuators
sigma = 2.4
m_acts = 6
actuator_x = np.linspace(8, 58, m_acts)       # gives x={8, 18, 28, 38, 48, 58}
actuator_y = np.linspace(8, 58, m_acts)       # gives y={8, 18, 28, 38, 48, 58} 

f = np.zeros_like(X)
for x in range(nx):
    for y in range(ny):
        for i in actuator_x:
            for j in actuator_y:
                f[x][y] += 1 / (2*np.pi*sigma**2) * np.e**( ((x-i)**2 + (y-j)**2) / (-2*sigma**2) )

# define initial conditions of field variable u
u0 = np.sin(np.pi*(X/Lx+Y/Ly)) + np.sin(np.pi*X/Lx) + np.sin(np.pi*Y/Ly)
#u0 = 2*(np.sin(np.pi*X/Lx) + np.sin(np.pi*Y/Ly))

# display initial conditions
fig, (u0_ax, R0_ax, f_ax) = plt.subplots(1, 3, figsize=(13, 4))
R = get_R(0, u0)
u0_ax.contourf(X, Y, u0)
R0_ax.contourf(X, Y, R)
f_ax.contourf(X, Y, f)
u0_ax.set_title("Initial u")
R0_ax.set_title("Initial R")
f_ax.set_title("Forcing actuators")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
Z = u0
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_proj_type('ortho')
ax.contour(X, Y, Z, levels=8, lw=3, linestyles="solid", offset=-4)
ax.set_xlim(0, 2*Lx)
ax.set_ylim(0, 2*Ly)
ax.set_zlim(-6, 6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("v(x,y)")
plt.show()

# run main function to get results
main(u0, 1e-8, 1e-8)