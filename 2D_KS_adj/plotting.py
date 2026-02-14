import numpy as np
import matplotlib.pyplot as plt
from get_R import get_R
from get_G import get_G
from input_vars import X, Y
from residual import compute_residuals

class Plotting:

    def plot_from_data(path):

        u = np.loadtxt(path)
        plt.contourf(X, Y, u, figsize=(7, 7))
        plt.show()

    ###############################################################################################

    def plot_initial(u0: np.ndarray[tuple[int, int], float]) -> None:

        # setup axis and figure
        fig, (u0_ax, R0_ax, G0_ax) = plt.subplots(1, 3, figsize=(15, 4))

        # obtain R and G fields
        R = get_R(0, u0)
        G = get_G(0, u0)

        # plot contours 
        u0_cont = u0_ax.contourf(X, Y, u0, antialised=True)
        R0_cont = R0_ax.contourf(X, Y, R, antialised=True)
        G0_cont = G0_ax.contourf(X, Y, G, antialised=True)

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

    def plot_final(u_lst: np.ndarray[tuple[int, int], float], t_lst) -> None:

        fig, (u_val, res) = plt.subplots(1, 2, figsize=(12, 5))
        
        # extract final u field
        u_final = u_lst[-1]
        np.nan_to_num(u_final, nan=0)

        # plot u field
        u_cont = u_val.contourf(X, Y, u_final, antialised=True, levels=7)
        u_val.set_xlabel('x')
        u_val.set_ylabel('y')
        fig.colorbar(u_cont)

        # plot residuals
        t_lst_trunc, G_lst_trunc = compute_residuals(t_lst, u_lst)
        res.plot(t_lst_trunc, G_lst_trunc)
        res.semilogy()
        res.set_xlabel('τ')
        res.set_title('Residual (RMS of G(u))')
        res.set_xlim(0, t_lst_trunc[-1])
        res.grid()

        plt.show()