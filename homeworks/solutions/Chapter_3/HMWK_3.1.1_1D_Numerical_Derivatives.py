import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

################# BEGIN USER INPUT #################
def get_globals():
    global x_GRID, Nx, DATA_DIR

    x_GRID = np.linspace( 0,10,1000 ) # START, END, NPOINTS   

    Nx = len( x_GRID )
    DATA_DIR = "1_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

def get_f_x():
    f_x   = x_GRID ** np.sin( x_GRID )
    return f_x    

def get_exact_deriv():
    return x_GRID ** ( -1 + np.sin( x_GRID ) ) *  (x_GRID * np.cos( x_GRID ) * np.log(x_GRID) + np.sin( x_GRID ) )

def get_NUMPY_deriv(f_x):
    return np.gradient(f_x,x_GRID)

def get_First_Order_Central_Derivative(x_GRID, f_x):

    # First-order finite difference (central)
    f_p_x = np.zeros(( Nx ))
    dx = x_GRID[1] - x_GRID[0]
    for xi in range( Nx ):
        if ( xi == 0 ):
            # Do forward difference for the first point,
            #   since there are no points behind
            f_p_x[xi]  = f_x[xi+1] - f_x[xi]
            f_p_x[xi] /= dx
        if ( xi < Nx-1 ):
            f_p_x[xi]  = f_x[xi+1] - f_x[xi-1]
            f_p_x[xi] /= 2 * dx
        if ( xi == Nx-1 ):
            # Do backward difference for the last point,
            #   since there are no points ahead
            f_p_x[xi]  = f_x[xi-1] - f_x[xi]
            f_p_x[xi] /= dx
    return f_p_x

def plot_func( x_GRID, f_x, f_p_x, title ):
    if ( len(x_GRID) < 30 ):
        NPLOT = 1
    else:
        NPLOT = len(x_GRID) // 30

    plt.plot(x_GRID,x_GRID*0,         "--",c='black',lw=1)
    plt.plot(x_GRID,f_x,            "-",c='black',lw=4,label="5*f(x)")
    plt.plot(x_GRID,get_exact_deriv(),"-",c='blue',lw=7,alpha=0.25,label="f'(x) (Exact)")
    plt.plot(x_GRID[0:-1:NPLOT//2],get_NUMPY_deriv(f_x)[0:-1:NPLOT//2],".",c='green',lw=4,label="f'(x) (NUMPY)")
    plt.plot(x_GRID[0:-1:NPLOT],f_p_x[0:-1:NPLOT]  ,"o",c='red',lw=2,label="f'(x) (1st-Order F)")
    plt.legend()
    plt.xlim(x_GRID[0],x_GRID[-1])
    plt.ylim(-10,10)
    plt.savefig(f"{DATA_DIR}/" + title + ".jpg",dpi=300)
    plt.clf()

def main():
    get_globals()
    f_x       = get_f_x()

    f_p_x  = get_First_Order_Central_Derivative(x_GRID, f_x)
    plot_func( x_GRID, f_x, f_p_x, "Error")

if ( __name__ == "__main__" ):
    main()




