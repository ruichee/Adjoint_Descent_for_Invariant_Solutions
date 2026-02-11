####################################################################################################

# Compilation of initial conditions that yielded converged results, both known and new solutions #
# Fourier coefficients (2 d.p.) are included, as well as required time steps and tolerances used #

####################################################################################################

################################### KNOWN SOLUTIONS FOUND ##########################################

# E2 found - 2366.97 0.0 0.0 0.0 0.0 1866.39 0.0 
'''u0 = np.cos(np.pi*X/Lx) + np.cos(np.pi*(-X/Lx + 2*Y/Ly)) + np.cos(np.pi*(-X/Lx - 2*Y/Ly))'''

# E7 found - 0.0 0.0 0.0 0.0 0.0 0.0 1292.97 
'''u0 = np.sin(2*np.pi*(X/Lx)) + np.sin(3*np.pi*(Y/Ly)) + np.cos(2*np.pi*(X/Lx+Y/Ly))'''

# E9 found - 0.0 0.0 0.0 -- 0.0 0.0 0.0 0.0 0.0 0.0 0.0 
'''u0 = np.sin(np.pi*(X/Lx + Y/Ly)) - np.sin(2*np.pi*(Y/Ly)) '''

# E10 found - 0.0 788.4 1869.96 0.0 0.0 788.4 0.0 
'''u0 = np.sin(np.pi*(-X/Lx + Y/Ly)) - np.sin(3*np.pi*(-X/Lx)) - np.cos(3*np.pi*(Y/Ly))'''

# E13 found - 2175.17 0.0 0.0 0.0 2175.17 0.0 901.21 
'''u0 = np.cos(2*np.pi*(Y/Ly)) + np.sin(2*np.pi*(X/Lx))'''
'''u0 = np.sin(np.sin(2*np.pi*(X/Lx)) + np.cos(2*np.pi*(Y/Ly)))'''

# E14 found - 0.0 0.0 0.0 0.0 5086.57 0.0 0.0 
'''u0 = np.sin(2*np.pi*Y/Ly) + np.sin(2*np.pi*(X/Lx - Y/Ly)) + np.sin(2*np.pi*(X/Lx + Y/Ly))'''

# E19 found - 0.0 0.0 301.96 -- 778.95 963.07 0.0 0.0 595.43 0.0 1020.05 
'''u0 = np.sin(np.pi*(X/Lx)) + np.sin(3*np.pi*(X/Lx)) + np.sin(2*np.pi*(Y/Ly)) '''

# E23 found - 0.0 0.0 2000.84 -- 2050.04 0.0 1060.15 0.0 2341.38 125.17 1208.1
'''u0 = np.cos(2*np.pi*(X/Lx)) + np.cos(2*np.pi*(Y/Ly)) + np.cos(3*np.pi*(X/Lx))'''

# E34 found - 0.0 404.94 1549.46 -- 368.01 615.94 170.16 778.19 826.38 198.7 146.1 
'''u0 = np.sin(np.pi*X/Lx) + np.sin(np.pi*(-2*X/Lx + Y/Ly)) + np.sin(np.pi*(-2*X/Lx - Y/Ly))'''

# E35 found - 0.0 419.25 0.0 -- 1491.4 0.0 0.0 856.84 1491.4 0.0 989.43 
'''u0 = np.sin(np.pi*(X/Lx - Y/Ly)) + np.sin(np.pi*(X/Lx + Y/Ly)) + np.sin(2*np.pi*X/Lx) + np.sin(2*np.pi*Y/Ly)'''

# E45 found - 0.0 797.04 0.0 -- 502.9 0.0 0.0 613.48 411.21 0.0 136.46 
'''u0 = np.sin(3*np.pi*Y/Ly) + np.sin(np.pi*(X/Lx - Y/Ly)) + np.sin(np.pi*(X/Lx + Y/Ly))'''

# E162 found - 609.02 189.9 606.6 -- 251.58 1006.11 974.24 229.92 251.03 1008.14 477.01
'''u0 = np.sin(np.pi*(-X/Lx + Y/Ly)) + np.sin(3*np.pi*(X/Lx)) - np.cos(2*np.pi*(Y/Ly))'''

# E229 found - 1398.75 416.66 0.0 -- 425.31 221.33 0.0 213.66 133.5 301.22 142.05
'''u0 = np.sin(2*np.pi*(X/Lx)) - np.sin(np.pi*(Y/Ly)) + np.sin(np.pi*(X/Lx - Y/Ly)) + np.sin(np.pi*(X/Lx + Y/Ly)) '''

# E231 found - 1459.68 1816.42 1459.68 -- 719.75 87.59 21.08 347.97 719.75 87.59 234.69 
'''u0 = np.sin(np.pi*(X/Lx + Y/Ly)) - np.sin(np.pi*(X/Lx - Y/Ly)) + np.sin(np.pi*(Y/Ly)) '''

# E234 found - 664.18 231.42 1469.74 -- 173.3 35.9 13.51 2.46 259.38 176.31 38.56 
'''u0 = np.cos(np.pi*(X/Lx)) - np.cos(2*np.pi*(Y/Ly)) + np.cos(3*np.pi*(X/Lx)) + np.cos(3*np.pi*(Y/Ly)) '''

# E248 found - 1628.9 0.0 0.0 -- 0.0 928.97 0.0 0.0 681.56 0.0 1137.79
'''u0 = np.sin(3*np.pi*Y/Ly) - np.sin(2*np.pi*(X/Lx - Y/Ly)) + np.sin(2*np.pi*(X/Lx + Y/Ly))'''

##################################### NEW SOLUTIONS FOUND ##########################################

# converging  new solution
'''u0 = np.sin(np.pi * X/Lx) + np.sin(np.pi * Y/Ly)'''

# another new solution - 0.0 0.0 0.0 485.41 193.83 0.0 0.0
'''u0 = np.sin(4* np.pi * X/Lx) + np.sin(2 * np.pi * X/Ly) + np.sin(3* np.pi * (Y/Ly+X/Lx) )'''

# new solution - 2727.57 475.75 2727.57 -- 145.15 195.94 885.87 536.15 145.15 195.94 429.09
'''u0 = np.cos(np.pi*(X/Lx)) + np.cos(np.pi*(Y/Ly)) + np.cos(3*np.pi*(X/Lx)) + np.cos(3*np.pi*(Y/Ly)) '''

# might be E285???
'''u0 = np.sin(np.pi*(X/Lx)) + np.cos(np.pi*(X/Lx)) + np.sin(np.pi*(Y/Ly)) + np.cos(np.pi*(Y/Ly)) '''

####################################################################################################