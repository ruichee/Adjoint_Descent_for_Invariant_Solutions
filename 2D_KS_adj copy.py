import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from scipy.optimize import newton_krylov


###############################################################################################

def get_vars(Lx: float, Ly: float, nx: int, ny: int) -> tuple[np.ndarray[any, float],...]:

    dx = Lx/nx                                  # define x spatial step
    dy = Ly/ny                                  # define x spatial step
    
    x = np.linspace(0, Lx-dx, nx)               # nx = EVEN no. of collocation points, define grid
    y = np.linspace(0, Ly-dy, ny)               # ny = EVEN no. of collocation points, define grid
    
    kx = 2*np.pi * np.fft.fftfreq(nx, d=Lx/nx)  # fourier wave numbers (kx) for DFT in x-dir
    ky = 2*np.pi * np.fft.fftfreq(ny, d=Ly/ny)  # fourier wave numbers (ky) for DFT in y-dir
    
    KX, KY = np.meshgrid(kx, ky)                # meshgrid of all combinations of kx and ky waves
    X, Y = np.meshgrid(x, y)                    # meshgrid of all combinations of x and y values
    
    return X, KX, Y, KY                         # NOTE: L-dx ensure no cutting into next period

###############################################################################################

def dealiase(ff) -> np.ndarray:
    
    global KX, KY

    kx_abs = np.absolute(KX)
    ky_abs = np.absolute(KY)

    kx_max = 2/3 * np.max(kx_abs)                       # maximum frequency that we will keep
    ky_max = 2/3 * np.max(ky_abs)                       # maximum frequency that we will keep

    ff_filterx = np.where(KX < kx_max, ff, 0)           # all higher frequencies in x are set to 0
    ff_filterxy = np.where(KY < ky_max, ff_filterx, 0)  # all higher frequencies in y are set to 0
    
    return ff_filterxy

###############################################################################################

def get_R(u: np.ndarray[tuple[int, int], float]) -> np.ndarray[tuple[int, int], float]: 

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

def get_G(t: float, u: np.ndarray[tuple[int, int], float], print_res=False) -> np.ndarray[tuple[int, int], float]:

    global KX, KY, f, step, nx, ny

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

    # add all terms together in fourier space
    G_f = np.fft.fft2(non_lin_term_1 + non_lin_term_2 + lin_term)
    G_f = dealiase(G_f)

    # set mean flow = 0, no DC component/offset
    mask = (KX==0) * (KY==0)
    G_f = np.where(mask, 0, G_f)

    # convert back to physical space
    G = np.real(np.fft.ifft2(G_f))

    # print to track iteration progress, use to check for sticking points
    if print_res:
        print(f"step: {step}, \t time: {t}, \t rms G: {np.linalg.norm(G) / np.sqrt(nx*ny)}, \t ||R||: {np.linalg.norm(R)}")

    return G

###############################################################################################

def steady_state_event(t, u):

    # rhs can either be get_R() or get_G()
    dudt = get_G(t, u)

    # compute R or G as a magnitude 
    change_in_u = np.linalg.norm(dudt) / np.sqrt(nx*ny)

    # set tolerance for ending iteration
    global u_tol
    tolerance = u_tol

    # compare, if G or R < tol, end iteration
    return change_in_u - tolerance

# Configure the Event
steady_state_event.terminal = True  # Stop the simulation when this event occurs
steady_state_event.direction = -1   # Only trigger when going from positive -> negative

###############################################################################################

def adj_descent_step(u0: np.ndarray[tuple[int, int], float], rtol: float, atol: float, T: int, dt: float, N: int) -> tuple[list, list]:

    global f, nx, ny

    # Set up the time interval
    nt = int(T / dt) + 1  
    tspan = np.linspace(0, T, nt)

    # Integration: use solve_ivp with method='BDF' (stiff system solver)
    solution = solve_ivp(
        fun=lambda t, u: 
            get_G(t, u.reshape(nx, ny), print_res=True)
            .flatten(),                     # function that returns du/dt
        t_span=(0, T),                      # (start_time, end_time)
        y0=u0.flatten(),                    # Initial condition
        method='BDF',                       # 'BDF' or 'Radau' - implicit + adaptive time stepping
        #events=steady_state_event,         # check if ||G(u)|| < tol, can end iteration early
        t_eval=tspan,                       # The specific time steps returned
        rtol=rtol,                          # Relative tolerance
        atol=atol                           # Absolute tolerance
    )

    # Extract the output list of iteration values
    u_lst = np.array([u.reshape(nx, ny) for u in solution.y.T])
    t_lst = solution.t.T

    u_new = newton_krylov(
        get_R, 
        u_lst[-1], 
        f_tol=1e-12,  
        method='gmres', 
        # 1. KILL THE LINE SEARCH: Force it to take the full Newton step
        line_search=None, 
        # 2. TIGHTEN THE INNER SOLVER: Force GMRES to be highly accurate
        inner_tol=1e-10,   
        # 3. INCREASE MAX INNER ITERATIONS: Give GMRES time to untangle the null space
        inner_maxiter=50, 
        iter=N,      
        verbose=True
    )

    return u_new

###############################################################################################

def compute_residuals(u_lst: list[np.ndarray[tuple[int, int], float]], steps: int = 1000):

    global nx, ny
    N = len(u_lst)

    # initialize list for residual values (RMS of G(u)) at each time step
    steps = min(steps, N)
    G_lst_trunc = np.zeros(steps+1)
    R_lst_trunc = np.zeros(steps+1)
    t_lst_trunc = range(steps+2)

    # iterate through u at each time step to find corresponding RMS of G(u), for just "steps" number of points
    for i in tqdm(range(steps)):
        G_lst_trunc[i] = np.linalg.norm(get_G(t_lst_trunc[i], u_lst[int(N/steps*i)])) / np.sqrt(nx*ny)
        R_lst_trunc[i] = np.linalg.norm(get_R(u_lst[int(N/steps*i)])) 
    
    # add last point as well
    R_lst_trunc[-1] = np.linalg.norm(get_R(u_lst[-1])) 
    G_lst_trunc[-1] = np.linalg.norm(get_G(t_lst_trunc[-1], u_lst[-1])) / np.sqrt(nx*ny)

    return t_lst_trunc, G_lst_trunc, R_lst_trunc

###############################################################################################

def plot_init(u0: np.ndarray[tuple[int, int], float]) -> None:

    # setup axis and figure
    fig, (u0_ax, R0_ax, G0_ax) = plt.subplots(1, 3, figsize=(15, 4))

    # obtain R and G fields
    R = get_R(u0)
    G = get_G(0, u0)

    # plot contours 
    u0_contlines = u0_ax.contour(X, Y, u0, colors="black", linewidths=1, linestyles="solid")
    u0_cont = u0_ax.contourf(X, Y, u0)
    R0_contlines = R0_ax.contour(X, Y, R, colors="black", linewidths=1, linestyles="solid")
    R0_cont = R0_ax.contourf(X, Y, R)
    G0_contlines = G0_ax.contour(X, Y, G, colors="black", linewidths=1, linestyles="solid")
    G0_cont = G0_ax.contourf(X, Y, G)

    # set titles and add colorbars
    u0_ax.set_title("Initial u")
    R0_ax.set_title("Initial R")
    G0_ax.set_title("Initial G")
    fig.colorbar(u0_cont)
    fig.colorbar(R0_cont)
    fig.colorbar(G0_cont)

    # display plots for u, G, R at initialization
    plt.show()

###############################################################################################

def plot_final(u_lst: np.ndarray[tuple[int, int], float]) -> None:

    fig, (u_val, res) = plt.subplots(1, 2, figsize=(12, 5))
    
    # extract final u field
    u_final = u_lst[-1]
    np.nan_to_num(u_final, nan=0)

    # plot u field
    u_cont = u_val.contourf(X, Y, u_final)
    u_contlines = u_val.contour(X, Y, u_final, linewidth=1, linestyle="solid", colors="black")
    u_val.set_xlabel('x')
    u_val.set_ylabel('y')
    fig.colorbar(u_cont)

    # plot residuals
    t_lst_trunc, G_lst_trunc, R_lst_trunc = compute_residuals(u_lst)
    res.plot(R_lst_trunc)
    res.semilogy()
    res.set_xlabel('τ')
    res.set_title('||R||')
    res.set_xlim(0, t_lst_trunc[-1])
    res.grid()

    plt.show()

###############################################################################################

def main(u0: np.ndarray[tuple[int, int], float], 
         steps: int, tols: float) -> None:

    # plot initial fields
    plot_init(u0)

    global T, N, dt, step

    u_prev = u0
    u_lst = [u0]

    u_new = adj_descent_step(u_prev, 1e-8, 1e-8, T=100, dt=dt, N=N)
    u_prev = u_new

    for i in range(steps):
        if i < len(tols):
            tol = tols[i]
        else:
            tol = tols[-1]
        
        step += 1
        u_new = adj_descent_step(u_prev, tol, tol, T=T, dt=dt, N=N)
        u_prev = u_new
        u_lst.append(u_new)

    print(len(u_lst))
    '''
    # Run 1
    u_lst1, t_lst1 = adj_descent(u0, tol1, tol1, T=T1, dt=1)
    
    # Run 2: starts from the end of run 1
    u_lst2, t_lst2 = adj_descent(u_lst1[-1], tol2, tol2, T=T2, dt=1)

    # Run 3: starts from the end of run 2
    u_lst3, t_lst3 = adj_descent(u_lst2[-1], tol3, tol3, T=T3, dt=1)

    # 1. Handle the Time Offset
    # Shift t_lst2 so it starts where t_lst1 ended
    t_lst2_shifted = t_lst2 + t_lst1[-1]
    t_lst3_shifted = t_lst3 + t_lst2[-1] + t_lst1[-1]

    # 2. Concatenate and Remove Overlap
    # Slicing [1:] removes the duplicate starting point (t=3) from the second run
    u_lst = np.concatenate((u_lst1, u_lst2[1:], u_lst3[1:]), axis=0)
    t_lst = np.concatenate((t_lst1, t_lst2_shifted[1:], t_lst3_shifted[1:]), axis=0)
    '''
    # extract final u field
    u_final = u_lst[-1]

    # check fourier values
    u_k = np.fft.fft2(u_final)
    func = lambda x,y: np.round(np.abs(u_k[x,y]), 2)
    print("\nFourier Coefficients")
    print(func(1, 0), func(1, 1), func(0, 1))
    print(f"e(2,0) e(2,1) e(3,0) e(3,1) e(0,2) e(1,2) e(2,2)")
    print(f"{func(0, 2)} {func(1, 2)} {func(0, 3)} {func(1, 3)} {func(2, 0)} {func(2, 1)} {func(2, 2)}")
    print(f"e(0,3): {func(3, 0)}, e(1,3): {func(3, 1)}")
    print()

    # plot final results 
    plot_final(u_lst)

    # save entire u_final array data to output_u.csv file
    np.savetxt('output_u.csv', u_final, delimiter=',', fmt='%.2f')

###############################################################################################

# define variables 
Lx, Ly = 10, 10                 # domain size
nx, ny = 64, 64                 # number of collocation points
dt = 1                          # only controls what interval we receive the output u_lst and t_lst to be (actual time step is controlled in solve_ivp)

# obtain domain field (x), and fourier wave numbers kx
X, KX, Y, KY = get_vars(2*Lx, 2*Ly, nx, ny)

# define initial conditions of field variable u
u0 = np.cos(np.pi*X/Lx) + np.cos(np.pi*(-X/Lx + 2*Y/Ly)) + np.cos(np.pi*(-X/Lx - 2*Y/Ly))

f = 0
#u0 = np.loadtxt("output_u.csv", delimiter=',')

# define iteration time variables
T = 500
N = 1
step = 0
tols = [1e-10] + [1e-12]*5 + [1e-14]*5 + [1e-16]

# call to main function to execute descent
main(u0, steps=10, tols=tols)