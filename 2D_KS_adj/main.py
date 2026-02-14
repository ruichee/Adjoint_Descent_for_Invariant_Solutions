import numpy as np
from adj_descent import adj_descent
from plotting import Plotting
import input_vars
from input_vars import stages, u0, dt


def main(u0: np.ndarray[tuple[int, int], float], 
         stages: tuple[tuple[int, float]], dt) -> None:

    # plot initial fields
    Plotting.plot_initial(u0)

    u_prev = u0
    u_lst = [u0]
    t_lst = [0]

    for T, tol in stages:
        input_vars.stage += 1
        u_lst1, t_lst1 = adj_descent(u_prev, tol, tol, T=T, dt=dt)
        u_prev = u_lst1[-1]

        t_lst1_shifted = t_lst1 + t_lst[-1] 
        u_lst = np.concatenate((u_lst, u_lst1[1:]), axis=0)
        t_lst = np.concatenate((t_lst, t_lst1_shifted[1:]), axis=0)

    # extract final u field
    u_final = u_lst[-1]

    # check fourier values
    u_k = np.fft.fft2(u_final)
    func = lambda x,y: np.round(np.abs(u_k[x,y]), 2)
    print("\nFourier Coefficients")
    print("\t", func(1, 0), func(1, 1), func(0, 1), "\n")
    print(f"\t e(2,0) e(2,1) e(3,0) e(3,1) e(0,2) e(1,2) e(2,2)")
    print(f"\t {func(0, 2)} {func(1, 2)} {func(0, 3)} {func(1, 3)} {func(2, 0)} {func(2, 1)} {func(2, 2)} \n")
    print(f"\t e(0,3): {func(3, 0)}, e(1,3): {func(3, 1)}")
    print()

    # plot final results 
    Plotting.plot_final(u_lst, t_lst)

    # save entire u_final array data to output_u.csv file
    np.savetxt(r'2D_KS_adj\fixed_points\output_u.dat', u_final, delimiter=',', fmt='%.2f')


if __name__ == "__main__":
    main(u0, stages, dt)