import numpy as np

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