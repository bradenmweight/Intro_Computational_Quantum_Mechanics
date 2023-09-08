import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import subprocess as sp

def get_Globals():
    global dt, A_0, B_0
    A_0                 = 1.0 # Initial concentration of reactant
    B_0                 = 0.0 # Initial concentration of product
    t_start             = 0.0
    t_end               = 20 # Reaction time
    dt                  = 0.01

    global k, kp
    # Reaction rate
    k  = 1.0
    kp = 0.5

    global t_grid, A_t, B_t
    t_grid  = np.arange( t_start, t_end+dt, dt ) 
    A_t = np.zeros( len(t_grid) ) # This will store the solution
    B_t = np.zeros( len(t_grid) ) # This will store the solution

    global DATA_DIR
    DATA_DIR = "5.1.2_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def do_Euler():
    # Set initial value
    A_t[0] = A_0 # n = 0
    B_t[0] = B_0 # n = 0

    for n in range( len(t_grid) - 1 ):

        # The current concentrations
        A_NOW    = A_t[n]
        B_NOW    = 1 - A_t[n]

        # The next concentrations
        A_t[n+1] = A_NOW + dt * ( -k * A_NOW + kp * B_NOW )
        B_t[n+1] = B_NOW + dt * (  k * A_NOW - kp * B_NOW )

def plot():

    plt.plot( t_grid, A_t[:], "-", lw=3, label="[A](t)")
    plt.plot( t_grid, B_t[:], "-", lw=3, label="[B](t)")
    plt.plot( t_grid, A_t[:] + B_t[:], "-", lw=3, label="[A](t) + [B](t)")
    plt.plot( t_grid, np.ones(len(t_grid)) * kp/k, "--", c="black", lw=3  )
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Reaction Time t",fontsize=15)
    plt.ylabel("Population",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t.jpg", dpi=400)
    plt.clf()


def main():
    get_Globals()
    do_Euler()
    plot()

if ( __name__ == "__main__" ):
    main()

