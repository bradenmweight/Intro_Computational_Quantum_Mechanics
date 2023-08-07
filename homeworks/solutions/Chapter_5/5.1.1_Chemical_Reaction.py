import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import subprocess as sp

def get_Globals():
    global dt, A_0
    A_0                 = 1.0 # Initial concentration of reactant
    t_start             = 0.0
    t_end               = 20 # Reaction time
    dt                  = 0.25

    global k
    # Reaction rate
    k  = 1.0

    global t_grid, A_t_Eu, A_t_LF, A_t_RK4
    t_grid  = np.arange( t_start, t_end+dt, dt ) 
    A_t_Eu = np.zeros( len(t_grid) ) # This will store the solution
    A_t_LF = np.zeros( len(t_grid) ) # This will store the solution
    A_t_RK4 = np.zeros( len(t_grid) ) # This will store the solution

    global DATA_DIR
    DATA_DIR = "5.1.1_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_ODE( A, t ):
    '''
    Return right-hand side of ODE: y'(t) = f( y(t), t )
    Here, f( y(t), t ) = k * y(t)
    INPUT: Value of function y (float), Value of time t (float)
    '''
    A_p_t = -k  * A
    
    return A_p_t

def do_Euler():
    # Set initial value
    A_t_Eu[0] = A_0 # n = 0

    for n in range( len(t_grid) - 1 ):
        A_t_Eu[n+1] = A_t_Eu[n] + dt * get_ODE( A_t_Eu[n], t_grid[n] )

def do_LF():
    # Set initial value
    A_t_LF[0] = A_0 # n = 0

    # Single step Euler
    A_t_LF[1] = A_t_LF[0] + dt * get_ODE( A_t_LF[0], t_grid[0] )

    # Do leap frog
    for n in range( 1, len(t_grid) - 1 ):
        A_t_LF[n+1] = A_t_LF[n-1] + 2 * dt * get_ODE( A_t_LF[n], t_grid[n] )


def do_RK4():
    # Set initial value
    A_t_RK4[0] = A_0 # n = 0

    for n in range( len(t_grid) - 1 ): # Start after initial time
        k1 = get_ODE( A_t_RK4[n],             t_grid[n]        )
        k2 = get_ODE( A_t_RK4[n] + k1 * dt/2, t_grid[n] + dt/2 )
        k3 = get_ODE( A_t_RK4[n] + k2 * dt/2, t_grid[n] + dt/2 )
        k4 = get_ODE( A_t_RK4[n] + k3 * dt  , t_grid[n] + dt   )

        A_t_RK4[n+1] = A_t_RK4[n] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def plot():


    plt.plot( t_grid, A_t_LF[:], ":", lw=1, label="LF")
    plt.plot( t_grid, A_t_Eu[:], "-", lw=3, label="Euler")
    plt.plot( t_grid, A_t_RK4[:], "--", lw=3, label="RK4")
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.ylim(0,1)
    plt.xlabel("Reaction Time t",fontsize=15)
    plt.ylabel("Population",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t_ALL.jpg", dpi=400)
    plt.clf()

    for step in range( len(t_grid) ):
        print( A_t_LF[step] )


    plt.plot( t_grid, A_t_Eu[:], "-", lw=3, label="[A](t)")
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Reaction Time t",fontsize=15)
    plt.ylabel("Population",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t_Eu.jpg", dpi=400)
    plt.clf()

    plt.plot( t_grid, A_t_LF[:], "-", lw=3, label="[A](t)")
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Reaction Time t",fontsize=15)
    plt.ylabel("Population",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t_LF.jpg", dpi=400)
    plt.clf()

    plt.plot( t_grid, A_t_Eu[:], "-", lw=3, label="[A](t)")
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Reaction Time t",fontsize=15)
    plt.ylabel("Population",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t_RK4.jpg", dpi=400)
    plt.clf()


def main():
    get_Globals()
    do_Euler()
    do_LF()
    do_RK4()
    plot()

if ( __name__ == "__main__" ):
    main()

