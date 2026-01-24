import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


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

    kx_max = 1/3 * np.max(kx_abs)                       # maximum frequency that we will keep
    ky_max = 1/3 * np.max(ky_abs)                       # maximum frequency that we will keep

    ff_filterx = np.where(KX < kx_max, ff, 0)           # all higher frequencies in x are set to 0
    ff_filterxy = np.where(KY < ky_max, ff_filterx, 0)  # all higher frequencies in y are set to 0
    
    return ff_filterxy

###############################################################################################

def get_R(u): 

    global KX, KY, f

    # obtain u in fourier space
    u_f = np.fft.fft2(u)                        # bring u into fourier
    u_f = dealiase(u_f)                         # dealise u

    # non-linear term -1/2(∂ₓu)^2 in fourier space 
    u_x_f = 1j * KX * u_f                       # ∂ₓu in fourier, differentiate via multiply ik_x
    u_y_f = 1j * KY * u_f                       # ∂ᵧu in fourier, differentiate via multiply ik_y
    u_x = np.fft.ifft2(u_x_f)                   # bring back to physical space
    u_y = np.fft.ifft2(u_y_f)                   # bring back to physical space
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
    
    return R

###############################################################################################

def get_G(t, u):

    global KX, KY, f

    # first obtain R and its fourier transform
    R = get_R(u)
    R_f = np.fft.fft2(R)

    # non-linear term (1) in fourier space
    u_f = np.fft.fft2(u)
    u_x_f = 1j * KX * u_f
    u_y_f = 1j * KY * u_f
    u_x = np.fft.ifft2(u_x_f)
    u_y = np.fft.ifft2(u_y_f)
    R_x_f = 1j * KX * R_f
    R_y_f = 1j * KY * R_f
    R_x = np.fft.ifft2(R_x_f)
    R_y = np.fft.ifft2(R_y_f)
    Rx_ux = R_x*u_x
    Ry_uy = R_y*u_y
    non_lin_term_1 = -Rx_ux -Ry_uy

    # non-linear term (2) in fourier space
    u_xx_f = -KX**2 * u_f
    u_yy_f = -KY**2 * u_f
    u_xx = np.fft.ifft2(u_xx_f)
    u_yy = np.fft.ifft2(u_yy_f)
    non_lin_term_2 = -R*u_xx -R*u_yy

    # linear terms in fourier space
    R_xx = np.fft.ifft2(-KX**2 * R_f)
    R_yy = np.fft.ifft2(-KY**2 * R_f)
    R_xxxx = np.fft.ifft2(KX**4 * R_f)
    R_yyyy = np.fft.ifft2(KY**4 * R_f)
    R_xxyy = np.fft.ifft2(KX**2 * KY**2 * R_f)
    lin_term = R_xx + R_yy + R_xxxx + R_yyyy + 2*R_xxyy

    G_f = np.fft.fft2(non_lin_term_1 + non_lin_term_2 + lin_term)
    G_f = dealiase(G_f)

    # set mean flow = 0, no DC component/offset
    mask = (KX==0) * (KY==0)
    G_f = np.where(mask, 0, G_f)

    G = np.real(np.fft.ifft2(G_f))

    print(f"time: {t}, norm: {np.linalg.norm(G)}")

    return G

###############################################################################################

def steady_state_event(t, u):

    # rhs can either be get_R() or get_G()
    dudt = get_G(u)

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

def adj_descent(u0, rtol, atol):

    global f, T, dt, nx, ny

    # Set up the time interval
    nt = int(T / dt) + 1  
    tspan = np.linspace(0, T, nt)

    # Integration: use solve_ivp with method='BDF' to mimic ode15s (stiff solver)
    solution = solve_ivp(
        fun=lambda t,u: \
            get_G(t, u.reshape(nx, ny))
            .flatten(),                     # function that returns du/dt
        t_span=(0, T),                      # (start_time, end_time)
        y0=u0.flatten(),                    # Initial condition
        method='BDF',                       # 'BDF' or 'Radau' - implicit adaptive time stepping
        #events=steady_state_event,         # check if ||G(u)|| < tol, can end iteration early
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

    for i in range(len(u_lst)):
        G_lst[i] = np.linalg.norm(get_G(t_lst[i], u_lst[i]))

    return G_lst

###############################################################################################

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
    
    G_lst = compute_residuals(u_lst)
    res.plot(t_lst, G_lst)
    res.semilogy()
    res.set_xlabel('τ')
    res.set_title('Residual of Adjoint Norm ||G(u)||')
    res.set_xlim(0, t_lst[-1])
    res.grid()

    fig.tight_layout()
    plt.show()

###############################################################################################

def main(u0, adj_rtol, adj_atol):

    u_lst, t_lst = adj_descent(u0, adj_rtol, adj_atol)

    fig, (u_val, res) = plt.subplots(1, 2, figsize=(10, 5))

    u_val.contourf(X, Y, u_lst[-1])

    G_lst = compute_residuals(t_lst, u_lst)
    res.plot(t_lst, G_lst)
    res.semilogy()
    res.set_xlabel('τ')
    res.set_title('Residual of Adjoint Norm ||G(u)||')
    res.set_xlim(0, t_lst[-1])
    res.grid()


    plt.show()


    # check fourier values
    u_k = np.fft.fft2(u_lst[-1])
    func = lambda x,y: np.round(np.abs(u_k[x,y]), 2)
    print(func(0, 1), func(1, 1), func(1, 0))
    print(func(2, 0), func(2, 1), func(3, 0), 
          func(3, 1), func(0, 2), func(1, 2), func(2, 2))

    # plot own results (integrated non-conservative form)
    #plot_data(u_lst, t_lst)
    
    return u_lst, t_lst

###############################################################################################

# define variables 
Lx, Ly = 20, 20                 # domain size
nx, ny = 64, 64                 # number of collocation points
T = 100                         # max iteration time
dt = 1                          # iteration step 
u_tol = 1e-6                    # tolerance for converged u

# obtain domain field (x), and fourier wave numbers kx
X, KX, Y, KY = get_vars(Lx, Ly, nx, ny)

# define initial conditions of field variable u
m = 1
n = 1
u0 = np.cos(2*np.pi*(n*Y/Ly)) + np.sin(2*np.pi*(m*X/Lx))
#u0 = np.sin(np.cos(2*np.pi*(m*X/Lx)) + np.cos(2*np.pi*(n*Y/Ly)))

#u0 = np.cos(2*np.pi*(n*Y/Ly + m*X/Lx)) - np.sin(np.cos(2*np.pi*(m*X/Lx))) - np.cos(np.cos(2*np.pi*(n*Y/Ly)))
f = 0 

# E13 found
'''m = 1
n = 1
u0 = np.cos(2*np.pi*(m*X/Lx + n*Y/Ly)) + np.sin(2*np.pi*m*X/Lx)
f = 0'''

# E13 found 
'''m = 1
n = 1
u0 = np.cos(2*np.pi*(n*Y/Ly)) + np.sin(2*np.pi*(m*X/Lx))
f = 0 '''

# potentially E223
'''m = 1
n = 1
u0 = np.sin(np.sin(2*np.pi*(m*X/Lx)) + np.cos(2*np.pi*(n*Y/Ly)))
f = 0'''

# display initial conditions
fig, (u0_ax, R0_ax, G0_ax) = plt.subplots(1, 3, figsize=(15, 5))
R = get_R(u0)
G = get_G(0, u0)
u0_ax.contourf(X, Y, u0)
R0_ax.contourf(X, Y, R)
G0_ax.contourf(X, Y, G)
u0_ax.set_title("Initial u")
R0_ax.set_title("Initial R")
G0_ax.set_title("Initial G")
plt.show()

# call to main function to execute descent
u_lst, t_lst = main(u0, adj_rtol=1e-8, adj_atol=1e-8)
