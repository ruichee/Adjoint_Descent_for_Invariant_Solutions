import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_R(u, f):
    pass


def get_G(u, f):
    R = get_R(u, f)
    pass


def adj_descent(u, f, dt, n_iter, tol):

    for _ in range(n_iter):

        un = u.copy()
        G = get_G(u, f)

        u = un + dt*G # can implement rk45 later on
        
        err: float # implement error calc
        if err < tol:
            break

    return u
        

def ngh_descent(u, f, dt, n_iter, tol):
    
    for _ in range(n_iter):

        un = u.copy()
        R = get_R(u, f)
        
        u = un + dt*R # can implement rk45 later on

        err: float # implement error calc
        if err < tol:
            break

    return u



def plot_data():
    pass


def main(u0, L, f, dt, n_iter_adj, n_iter_ngh, tol_adj, tol_ngh):
    

    u = adj_descent(u0, f, dt, n_iter_adj, tol_adj)
    
    # check if ngh descent is required here

    u = ngh_descent(u0, f, dt, n_iter_ngh, tol_ngh)

    plot_data()


