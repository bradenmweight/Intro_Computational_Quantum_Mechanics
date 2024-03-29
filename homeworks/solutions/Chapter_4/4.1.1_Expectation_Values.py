import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "4.1.1_PLOTS"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_f_x():
    global p0_LIST
    p0_LIST = np.array([0,1,2,3,4,5])
    Nx = 1000
    
    x = np.linspace( -20,20,Nx )
    f_x = np.zeros( (len(p0_LIST), Nx), dtype=complex )
    for p0IND,p0 in enumerate( p0_LIST ):
        f_x[p0IND,:] = np.exp( 1.j * 2 * np.pi * p0 * x ) * np.exp( -x**2 / 2 )

    return x, f_x

def plot_f_x(x,f_x):

    color_list = ['black','red','green','blue','orange','purple']
    for p0IND,p0 in enumerate( p0_LIST[:2] ):
        if ( p0IND == 0 ):
            plt.plot( x, f_x[p0IND].real, "--", lw=3, c=color_list[p0IND], label=f"p0 = {round(p0,1)} RE" )
            plt.plot( x, f_x[p0IND].imag, ":", lw=3, c=color_list[p0IND], label=f"p0 = {round(p0,1)} IM" )
            plt.plot( x, np.abs(f_x[p0IND]), "-", lw=3, c=color_list[p0IND], label=f"p0 = {round(p0,1)} ABS" )
        else:
            plt.plot( x, f_x[p0IND].real, "--", lw=2, c=color_list[p0IND] )
            plt.plot( x, f_x[p0IND].imag, ":", lw=2, c=color_list[p0IND] )
            plt.plot( x, np.abs(f_x[p0IND]), "-", lw=1, c=color_list[p0IND], label=f"p0 = {round(p0,1)}" )

    plt.legend()
    plt.xlim(-4,4)
    plt.xlabel("x",fontsize=15)
    plt.ylabel("Wavefunction",fontsize=15)
    plt.savefig(f"{DATA_DIR}/f_x.jpg",dpi=400)
    plt.clf()

def get_PROB( f ):
    return np.real( np.conjugate(f) * f )

def norm( x,f_x ):
    dx = x[1] - x[0]
    for p0IND,p0 in enumerate( p0_LIST ):
        PROB  = get_PROB( f_x[p0IND] )
        f_x[p0IND,:] /= np.sqrt( np.sum(PROB) * dx )
        PROB  = get_PROB( f_x[p0IND] )
    
    return x, f_x

def get_f_p( x, f_x ):

    dx = x[1] - x[0]
    Lx = x[1] - x[0]
    Nx = len(x)

    # Define the p-grid (reciprocal grid)
    dp   = 2 * np.pi / Lx
    pmax = 1 / 2 / dx # This is not angular frequency
    p    = np.linspace( -pmax, pmax, Nx )

    f_p = np.zeros( (len(p0_LIST), Nx), dtype=complex )

    # Define the Fourier matrix, W, and operate
    n = np.arange(Nx, dtype=complex).reshape( (-1,1) )
    m = np.arange(Nx, dtype=complex).reshape( (1,-1) )
    W = np.exp( -2j*np.pi * (m-Nx//2) * (n-Nx//2) / Nx )

    for p0IND,p0 in enumerate( p0_LIST ):
        f_p[p0IND,:] = W @ f_x[p0IND,:]
        f_p[p0IND,:] *= dx / np.sqrt( 2 * np.pi )

    return p, f_p

def plot_f_p(p,f_p):

    color_list = ['black','red','green','blue','orange','purple']
    for p0IND,p0 in enumerate( p0_LIST[:5] ):
        if ( p0IND == 0 ):
            plt.plot( p, f_p[p0IND].real, "--",    c=color_list[p0IND], label=f"$p_0$ = {round(p0,1)} RE" )
            plt.plot( p, f_p[p0IND].imag, ":",     c=color_list[p0IND], label=f"$p_0$ = {round(p0,1)} IM" )
            plt.plot( p, np.abs(f_p[p0IND]), "-",  c=color_list[p0IND], label=f"$p_0$ = {round(p0,1)} ABS" )
        else:
            plt.plot( p, f_p[p0IND].real, "--",    c=color_list[p0IND], label=f"$p_0$ = {round(p0,1)}" )
            plt.plot( p, f_p[p0IND].imag, ":",     c=color_list[p0IND] )
            plt.plot( p, np.abs(f_p[p0IND]), "-",  c=color_list[p0IND] )
    plt.legend()
    plt.xlim(p0_LIST[0]-2,p0_LIST[-1]+2)
    plt.xlabel("p",fontsize=15)
    plt.ylabel("Wavefunction",fontsize=15)
    plt.savefig(f"{DATA_DIR}/f_p.jpg",dpi=400)
    plt.clf()

def get_observables( x,f_x,p,f_p ):

    dx = x[1] - x[0]
    dp = p[1] - p[0]
    
    # Define the conjugate of each for each p (implicit)
    f_x_conj = np.conjugate( f_x[:,:] )
    f_p_conj = np.conjugate( f_p[:,:] )

    # Define observables
    X_AVE  = np.zeros( (len(p0_LIST)) )
    X2_AVE = np.zeros( (len(p0_LIST)) )
    P_AVE  = np.zeros( (len(p0_LIST)) )
    P2_AVE = np.zeros( (len(p0_LIST)) )

    for p0IND,p0 in enumerate( p0_LIST ):
        # Do expectations of position
        PROB   = get_PROB( f_x[p0IND] )
        X_AVE[p0IND]  = np.sum( PROB * x    ) * dx
        X2_AVE[p0IND] = np.sum( PROB * x**2 ) * dx
        
        # Do expectations of momentum
        PROB   = get_PROB( f_p[p0IND] )
        P_AVE[p0IND]  = np.sum( PROB * p    ) * dp
        P2_AVE[p0IND] = np.sum( PROB * p**2 ) * dp

    np.savetxt( f"{DATA_DIR}/EXPECTATION_VALUES.dat", np.c_[ p0_LIST, X_AVE, X2_AVE, P_AVE, P2_AVE ], fmt="%1.4f" )

    return X_AVE, X2_AVE, P_AVE, P2_AVE

def plot_OBSERVABLES( X_AVE, X2_AVE, P_AVE, P2_AVE ):

    plt.plot( p0_LIST, X_AVE,"-o", label="$\langle X \\rangle$" )
    plt.plot( p0_LIST, X2_AVE,"-o", label="$\langle X^2 \\rangle$" )
    plt.legend()
    plt.savefig(f"{DATA_DIR}/X_X2.jpg",dpi=400)
    plt.clf()

    plt.plot( p0_LIST, np.round(X2_AVE - X_AVE**2,5),"-o", label="$\langle (\Delta X)^2 \\rangle$" )
    plt.legend()
    plt.title("Constant $\langle (\Delta X)^2 \\rangle$ implies that the width of f(x) is constant.",fontsize=15)
    plt.savefig(f"{DATA_DIR}/X_VAR.jpg",dpi=400)
    plt.clf()

    plt.plot( p0_LIST, P_AVE,"-o", label="$\langle P \\rangle$" )
    plt.legend()
    plt.savefig(f"{DATA_DIR}/P.jpg",dpi=400)
    plt.clf()

    plt.plot( p0_LIST, P2_AVE,"-o", label="$\langle P^2 \\rangle$" )
    plt.legend()
    plt.savefig(f"{DATA_DIR}/P2.jpg",dpi=400)
    plt.clf()

    plt.plot( p0_LIST, np.round(P2_AVE - P_AVE**2,5),"-o", label="$\langle (\Delta P)^2 \\rangle$" )
    plt.legend()
    plt.title("Constant $\langle (\Delta P)^2 \\rangle$ implies that the width of f(p) is constant.",fontsize=15)
    plt.savefig(f"{DATA_DIR}/P_VAR.jpg",dpi=400)
    plt.clf()


def main():
    x, f_x = get_f_x()
    x, f_x = norm(x,f_x)
    plot_f_x(x,f_x)
    p, f_p = get_f_p(x,f_x)
    p, f_p = norm(p, f_p)
    plot_f_p(p,f_p)
    X_AVE, X2_AVE, P_AVE, P2_AVE = get_observables( x,f_x,p,f_p )
    plot_OBSERVABLES( X_AVE, X2_AVE, P_AVE, P2_AVE )

if ( __name__ == "__main__" ):
    main()
