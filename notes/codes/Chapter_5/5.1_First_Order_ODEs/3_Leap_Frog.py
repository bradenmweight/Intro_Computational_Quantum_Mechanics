import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "3_PLOTS_DATA"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global t_grid, dt, y_0
    y_0                 = 0.0 # Initial value
    t_start             = 0.0
    t_end               = 1.0
    dt                  = 0.01 # Try smaller and larger for this

    global y_t_EULER, y_t_LEAP_FROG
    t_grid        = np.arange( t_start, t_end+dt, dt ) 
    y_t_EULER     = np.zeros( len(t_grid) ) # This will store the solution
    y_t_LEAP_FROG = np.zeros( len(t_grid) ) # This will store the solution

def get_y_p_t( y, t ):
    '''
    Return right-hand side of ODE: y'(t) = f( y(t), t )
    Here, f( y(t), t ) = e^(-y(t))
    INPUT: Value of function y (float), Value of time t (float)
    '''
    y_p_t = np.exp( -y ) # Time t does not appear in this example
    return y_p_t

def do_Forward_Euler():
    # Set initial value
    y_t_EULER[0] = y_0 # n = 0

    for n in range( len(t_grid) - 1 ): # Start after initial time
        y_t_EULER[n+1] = y_t_EULER[n] + dt * get_y_p_t( y_t_EULER[n], t_grid[n] )

def do_Leap_Frog():
    # Set initial value
    y_t_LEAP_FROG[0] = y_0 # n = 0

    # Do single step of Forward Euler to get n = 1
    y_t_LEAP_FROG[1] = y_t_LEAP_FROG[0] + dt * get_y_p_t( y_t_LEAP_FROG[0], t_grid[0] )

    for n in range( 1, len(t_grid) - 1 ): # Start after initial time
        y_t_LEAP_FROG[n+1] = y_t_LEAP_FROG[n-1] + 2 * dt * get_y_p_t( y_t_LEAP_FROG[n], t_grid[n] )

def get_analytic():
    return np.log( 1 + t_grid ) # y_0 = 0

def plot_y_t():

    plt.plot( t_grid, y_t_EULER, "-", lw=3, label="Forward Euler (Explicit)" )
    plt.plot( t_grid, y_t_LEAP_FROG, ":", lw=3, label="leap Frog (Explicit)" )
    plt.plot( t_grid, get_analytic(), "--", lw=3, label="ANALYTIC: Log[1+t]" )
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Time t",fontsize=15)
    plt.ylabel("y(t)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/y_t.jpg", dpi=400)
    plt.clf()

def plot_error():

    plt.plot( t_grid, get_analytic() - y_t_EULER, "-", lw=3, label="ANALYTIC - Forward Euler (Explicit)" )
    plt.plot( t_grid, get_analytic() - y_t_LEAP_FROG, "--", lw=3, label="ANALYTIC - Leap Frog (Implicit)" )
    plt.legend()
    plt.xlim(t_grid[0], t_grid[-1])
    plt.xlabel("Time t",fontsize=15)
    plt.ylabel("y(t)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/ERROR.jpg", dpi=400)
    plt.clf()

def main():
    get_Globals()
    do_Forward_Euler()
    do_Leap_Frog()
    plot_y_t()
    plot_error()

if ( __name__ == "__main__" ):
    main()

