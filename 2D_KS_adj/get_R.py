import numpy as np
from dealiase import dealiase
from input_vars import KX, KY, f

def get_R(t, u: np.ndarray[tuple[int, int], float], print_res=False) -> np.ndarray[tuple[int, int], float]: 

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

    # print to track iteration progress, use to check for sticking points
    if print_res:
        print(f"time: {t}, \t ||R||: {np.linalg.norm(R)}")
    
    return R