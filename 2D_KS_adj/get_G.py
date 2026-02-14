import numpy as np
import input_vars
from get_R import get_R
from dealiase import dealiase
from input_vars import stage, nx, ny, KX, KY, f

def get_G(t: float, u: np.ndarray[tuple[int, int], float],
          print_res=False) -> np.ndarray[tuple[int, int], float]:

    global stage, nx, ny, KX, KY, f

    # first obtain R and its fourier transform
    R = get_R(t, u)
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
        print(f"stage: {input_vars.stage}, \t time: {t}, \t norm: {np.linalg.norm(G) / np.sqrt(nx*ny)}")

    return G