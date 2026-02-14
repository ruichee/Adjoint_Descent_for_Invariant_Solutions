import numpy as np
from tqdm import tqdm
from get_G import get_G
from input_vars import nx, ny

def compute_residuals(t_lst: list, u_lst: list[np.ndarray[tuple[int, int], float]], steps: int = 1000):

    global nx, ny
    N = len(u_lst)

    # initialize list for residual values (RMS of G(u)) at each time step
    steps = min(steps, N)
    G_lst_trunc = np.zeros(steps+1)
    t_lst_trunc = np.zeros(steps+1)

    # iterate through u at each time step to find corresponding RMS of G(u), for just "steps" number of points
    for i in tqdm(range(steps)):
        G_lst_trunc[i] = np.linalg.norm(get_G(t_lst[int(N/steps*i)], u_lst[int(N/steps*i)])) / np.sqrt(nx*ny)
        t_lst_trunc[i] = t_lst[int(N/steps*i)]
    
    # add last point as well
    G_lst_trunc[-1] = np.linalg.norm(get_G(t_lst[-1], u_lst[-1])) / np.sqrt(nx*ny)
    t_lst_trunc[-1] = t_lst[-1]

    return t_lst_trunc, G_lst_trunc