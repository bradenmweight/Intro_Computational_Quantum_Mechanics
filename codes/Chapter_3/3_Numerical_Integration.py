import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from scipy import integrate


################# BEGIN USER INPUT #################
def get_globals():
    global DATA_DIR

    DATA_DIR = "3_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

def get_f_x( x ):
        """
        'x' can be number or array
        """
        return x ** 2

def get_exact_integral():
    return 1/3

################# END USER INPUT #################

#def get_NUMPY_deriv(f_x):
#    return np.gradient(f_x,GRID)

def get_Riemann_Integral(f_x, GRID):
    # For-loop version
    INTEGRAL = 0.0
    Nx = len(GRID)
    dx = GRID[1] - GRID[0]
    for xi,x in enumerate(GRID):
        INTEGRAL += f_x[ xi ] * dx
    """
    # Fast version
    INTEGRAL = np.sum( f_x[:] ) * dx
    """
    return INTEGRAL

def get_trapazoidal_Integral(f_x, GRID):
    Nx = len(GRID)
    dx = GRID[1] - GRID[0]
    INTEGRAL = 0.0
    for xi,x in enumerate(GRID):
        if ( xi < Nx-1 ):
            INTEGRAL += 0.5 * (f_x[ xi ] + f_x[ xi+1 ] ) * dx
    return INTEGRAL

def test_methods():
    # Let's test all the methods once
    Nx = 100
    GRID = np.linspace( 0,1,Nx ) # START, END, NPOINTS   

    f_x   = get_f_x( GRID )
    I_R   = get_Riemann_Integral(f_x, GRID)
    I_T   = get_trapazoidal_Integral(f_x, GRID)
    
    print("\nManual versions:")
    print("\tRIEMANN      :", I_R)
    print("\tTRAPAZOIDAL  :", I_T)
    
    print("\nSCIPY Versions:")
    print("\tTRAPAZOIDAL :", integrate.trapezoid( f_x, GRID ) )
    print("\tSIMPSON     :", integrate.simpson( f_x, GRID ) )
    print("\tQUAD        :", integrate.quad( get_f_x, GRID[0], GRID[-1] )[0], "Error: %1.3e" % (integrate.quad( get_f_x, GRID[0], GRID[1] )[1]) )

    print("\n\tEXACT     :", get_exact_integral())

def get_convergence_plot():

    ###########################
    # Let's run for a bunch of different grid densities and check error
    #del Nx, GRID # Remove global variables

    Nx_LIST = np.logspace( 1,6,endpoint=True,dtype=int ) # [10**start, 10**stop]
    RIEM    = np.zeros(( len(Nx_LIST) ))
    TRAP_MAN    = np.zeros(( len(Nx_LIST) ))
    TRAP    = np.zeros(( len(Nx_LIST) ))
    SIMP    = np.zeros(( len(Nx_LIST) ))
    QUAD    = np.zeros(( len(Nx_LIST) ))
    for n,N in enumerate( Nx_LIST ):
            GRID = np.linspace( 0,1,N ) # START, END, NPOINTS 
            f_x         = get_f_x( GRID )
            RIEM[n]     = get_Riemann_Integral(f_x, GRID)
            TRAP_MAN[n] = get_trapazoidal_Integral(f_x, GRID)
            TRAP[n]     = integrate.trapezoid( f_x, GRID )
            if ( N%2 == 0 ): 
                 SIMP[n] = integrate.simpson( f_x, GRID )
            else:
                 SIMP[n] = None
            QUAD[n] = integrate.quad( get_f_x, GRID[0], GRID[-1] )[0]

    plt.loglog(Nx_LIST, np.abs(RIEM-get_exact_integral()), "-", lw=3, c="black", label="RIEM (Man.)")
    plt.loglog(Nx_LIST, np.abs(TRAP_MAN-get_exact_integral()), "-", lw=6, alpha=0.5, c="red", label="TRAP (Man.)")
    plt.loglog(Nx_LIST, np.abs(TRAP-get_exact_integral()), "--", lw=3, c="red", label="TRAP (SCIPY)")
    plt.loglog(Nx_LIST, np.abs(SIMP-get_exact_integral()), "o", lw=3, c="blue", label="SIMP (SCIPY)")
    plt.loglog(Nx_LIST, np.abs(QUAD-get_exact_integral()), "-", lw=3, c="green", label="QUAD (SCIPY)")
    plt.legend(loc='upper right')
    plt.xlabel("Number of Function Points, Nx",fontsize=15)
    plt.ylabel("$|X_{APPROX} - X_{EXACT}|$",fontsize=15)
    plt.savefig(f"{DATA_DIR}/CONVERGENCE.jpg", dpi=600)


def main():
    get_globals()
    test_methods()
    get_convergence_plot()



if ( __name__ == "__main__" ):
    main()