import numpy as np
from scipy.integrate import solve_ivp
from get_G import get_G
from input_vars import nx, ny, f

def adj_descent(u0: np.ndarray[tuple[int, int], float], rtol: float, atol: float, T: int, dt: float) -> tuple[list, list]:

    global nx, ny, f

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
        t_eval=tspan,                       # The specific time steps returned
        rtol=rtol,                          # Relative tolerance
        atol=atol                           # Absolute tolerance
    )

    # Extract the output list of iteration values
    u_lst = np.array([u.reshape(nx, ny) for u in solution.y.T])
    t_lst = solution.t.T
    
    return u_lst, t_lst
