import numpy as np
from input_vars import KX, KY

def dealiase(ff) -> np.ndarray:

    kx_abs = np.absolute(KX)
    ky_abs = np.absolute(KY)

    kx_max = 2/3 * np.max(kx_abs)                       # maximum frequency that we will keep
    ky_max = 2/3 * np.max(ky_abs)                       # maximum frequency that we will keep

    ff_filterx = np.where(KX < kx_max, ff, 0)           # all higher frequencies in x are set to 0
    ff_filterxy = np.where(KY < ky_max, ff_filterx, 0)  # all higher frequencies in y are set to 0
    
    return ff_filterxy
