import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp


################# BEGIN USER INPUT #################
def get_globals():
    global x_GRID, Nx, DATA_DIR

    x_GRID = np.linspace( 0,1,100 ) # START, END, NPOINTS   

    Nx = len( x_GRID )
    DATA_DIR = "1_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)


def get_f_x():
    f_x   = x_GRID**5 - x_GRID**3
    #f_x   = np.sin(x_GRID*2*np.pi)**2
    #f_x   = (x_GRID-0.5) * np.exp( -(x_GRID - 0.5)**2 / 0.01 )
    return f_x    

def get_exact_deriv():
    return 5*x_GRID**4 - 3 * x_GRID**2
    return 2*np.sin(x_GRID*2*np.pi)*np.cos(x_GRID*2*np.pi)*2*np.pi
    """
    return np.exp( -(x_GRID - 0.5)**2 / 0.01 ) \
           + (x_GRID-0.5) * (-2*(x_GRID - 0.5) / 0.01) \
           * np.exp( -(x_GRID - 0.5)**2 / 0.01 )
    """
################# END USER INPUT #################



def get_NUMPY_deriv(f_x):
    return np.gradient(f_x,x_GRID)

def get_First_Order_Forward_Derivative(x_GRID, f_x):

    # First-order finite difference (forward)
    f_p_x = np.zeros(( Nx ))
    dx = x_GRID[1] - x_GRID[0]
    for xi in range( Nx ):
        if ( xi < Nx-1 ):
            f_p_x[xi]  = f_x[xi+1] - f_x[xi]
            f_p_x[xi] /= dx
        if ( xi == Nx-1 ):
            # Do backward difference for the last point,
            #   since there are no more points ahead
            f_p_x[xi]  = f_x[xi-1] - f_x[xi]
            f_p_x[xi] /= dx
    return f_p_x


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
    plt.plot(x_GRID,5*f_x,            "-",c='black',lw=4,label="5*f(x)")
    plt.plot(x_GRID,get_exact_deriv(),"-",c='blue',lw=7,alpha=0.25,label="f'(x) (Exact)")
    plt.plot(x_GRID[0:-1:NPLOT//2],get_NUMPY_deriv(f_x)[0:-1:NPLOT//2],".",c='green',lw=4,label="f'(x) (NUMPY)")
    plt.plot(x_GRID[0:-1:NPLOT],f_p_x[0:-1:NPLOT]  ,"o",c='red',lw=2,label="f'(x) (1st-Order F)")
    plt.legend()
    plt.xlim(x_GRID[0],x_GRID[-1])
    plt.savefig(f"{DATA_DIR}/" + title + ".jpg",dpi=300)
    plt.clf()

def plot_func_diff( x_GRID, f_x, f_p_x_1F, f_p_x_1C, title ):
    DIFF_1F = get_exact_deriv()-f_p_x_1F
    DIFF_1C = get_exact_deriv()-f_p_x_1C
    
    plt.plot(x_GRID,x_GRID*0,         "--",c='black',lw=1)
    plt.plot(x_GRID[:-1],DIFF_1F[:-1],"-",c='black',lw=6,alpha=0.25,label="First-Order Forward (dx)")
    plt.plot(x_GRID[:-1],DIFF_1C[:-1],"--",c='red',lw=4,label="First-Order Central (dx$^2$)")
    plt.legend()
    plt.xlim(x_GRID[0],x_GRID[-1])
    plt.title("f'(x) (Exact) - f'(x) (Approx)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/" + title + ".jpg",dpi=300)
    plt.clf()

def main():
    get_globals()
    f_x       = get_f_x()

    f_p_x_1F  = get_First_Order_Forward_Derivative(x_GRID, f_x)
    plot_func(x_GRID, f_x, f_p_x_1F, "First_Order_Forward")

    f_p_x_1C  = get_First_Order_Central_Derivative(x_GRID, f_x)
    plot_func_diff(x_GRID, f_x, f_p_x_1F, f_p_x_1C, "Error")

if ( __name__ == "__main__" ):
    main()