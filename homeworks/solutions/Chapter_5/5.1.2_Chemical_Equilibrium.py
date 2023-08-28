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
    k  = 0.5
    kp = 0.1

    global t_grid, A_t, B_t
    t_grid  = np.arange( t_start, t_end+dt, dt ) 
    A_t = np.zeros( len(t_grid) ) # This will store the solution
    B_t = np.zeros( len(t_grid) ) # This will store the solution

    global DATA_DIR
    DATA_DIR = "5.1.2_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_ODEs( A, B, t ):
    '''
    Return right-hand side of ODE: y'(t) = f( y(t), t )
    Here, f( y(t), t ) = k * y(t)
    INPUT: Value of function y (float), Value of time t (float)
    '''
    # A_p_t = -k  * A
    # B_p_t = -kp * (1 - A)
    
    A_p_t = -k  * A + kp * (1 - A)
    B_p_t = 1

    return A_p_t, B_p_t

def do_Euler():
    # Set initial value
    A_t[0] = A_0 # n = 0
    B_t[0] = B_0 # n = 0

    for n in range( len(t_grid) - 1 ):

        A0 = A_t[n]
        B0 = 1 - A_t[n]
        A_t[n+1] = A_t[n] + dt * ( -k * A0 + kp * B0 )
        B_t[n+1] = B_t[n] + dt * (  k * A0 - kp * B0 )


def do_RK4():
    # Set initial value
    A_t[0] = A_0 # n = 0
    B_t[0] = B_0 # n = 0

    for n in range( len(t_grid) - 1 ): # Start after initial time
        A1, B1 = get_ODEs( A_t[n],             B_t[n],             t_grid[n]        )
        A2, B2 = get_ODEs( A_t[n] + A1 * dt/2, B_t[n] + B1 * dt/2, t_grid[n] + dt/2 )
        A3, B3 = get_ODEs( A_t[n] + A2 * dt/2, B_t[n] + B2 * dt/2, t_grid[n] + dt/2 )
        A4, B4 = get_ODEs( A_t[n] + A3 * dt  , B_t[n] + B3 * dt,   t_grid[n] + dt   )

        A_t[n+1] = A_t[n] + dt * (A1 + 2 * A2 + 2 * A3 + A4) / 6
        B_t[n+1] = B_t[n] + dt * (B1 + 2 * B2 + 2 * B3 + B4) / 6


def plot():

    plt.plot( t_grid, A_t[:], "-", lw=3, label="[A](t)")
    plt.plot( t_grid, B_t[:], "-", lw=3, label="[B](t)")
    plt.plot( t_grid, A_t[:] + B_t[:], "-", lw=3, label="Total")
    plt.plot( t_grid, A_t[:] + B_t[:], "-", lw=3, label="Total")
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Reaction Time t",fontsize=15)
    plt.ylabel("Population",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t.jpg", dpi=400)
    plt.clf()


def main():
    get_Globals()
    do_Euler()
    #do_RK4()
    plot()

if ( __name__ == "__main__" ):
    main()

