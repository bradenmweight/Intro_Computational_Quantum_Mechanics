import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "7.5.1_QHO_by_Variation"
sp.call( "mkdir -p %s" % DATA_DIR, shell=True )

def get_Globals():

    global SIG_initial, d_SIG
    SIG_initial = 5.0 # psi_T = np.exp( -x^2 / 2 / SIG**2 )
    d_SIG       = 1e-5

    global xGRID, dx, Nx
    xMIN  = -10.0
    xMAX  =  10.0
    Nx    =  1001
    xGRID = np.linspace( xMIN, xMAX, Nx )
    dx    = xGRID[1] - xGRID[0]

def get_Vx( x ):
    Vx = 0.500 * x**2
    return Vx

def get_Vx_op():
    return np.diag( get_Vx(xGRID) )

def get_Tx():
    Tx = np.zeros( (Nx,Nx) )
    for i in range(Nx):
        for j in range(Nx):
            if ( i == j ):
                Tx[i,j] = np.pi**2 / 3.0
            else:
                Tx[i,j] = (-1.0)**(i-j) * 2 / (i-j)**2
    return Tx / 2 / dx**2


def psi( SIG ):
    return np.exp( -xGRID**2 / 2 / SIG**2 )

def get_energy( psi_T, H ):
    return np.einsum("x,xy,y->", psi_T, H, psi_T ) / np.einsum("x,x->", psi_T, psi_T )

def get_Gradient( SIG, H ):
    E_plus = get_energy( psi( SIG + d_SIG ), H )
    E_minus = get_energy( psi( SIG - d_SIG ), H )
    return ( E_plus - E_minus ) / 2 / d_SIG

def main():
    get_Globals()

    H               = get_Tx() + get_Vx_op()
    E_EXACT,U_EXACT = np.linalg.eigh( H )
    print( "Exact GS Energy: %1.6f" % E_EXACT[0] )



    # Record initial trial wavefunction and energy based on user input SIG
    SIG_old = SIG_initial
    psi_T     = psi( SIG_old )
    E_old     = get_energy( psi_T, H )

    SIG_LIST = [SIG_old]
    psi_T_LIST = [psi_T]
    E_LIST     = [E_old]

    # Perform gradient descent method to find the optimal SIG
    while True:
        SIG_new = SIG_old - 0.1 * get_Gradient( SIG_old, H )
        psi_T     = psi( SIG_new )
        E_new     = get_energy( psi_T, H )

        conv_SIG = np.abs( SIG_new - SIG_old ) / np.abs( SIG_new )
        conv_E     = np.abs( E_new - E_old ) / np.abs( E_new )
        if ( conv_SIG < 1.0e-8 and conv_E < 1.0e-8 ):
            break

        print( "SIG: %1.10f, Energy: %1.10f" % ( SIG_new, E_new ) )

        # Prepare for next iteration
        SIG_old = SIG_new
        E_old     = E_new

        # Save data for plotting
        SIG_LIST  .append( SIG_new )
        psi_T_LIST.append( psi_T )
        E_LIST    .append( E_new )

    print( "\nConvergence Reached." )
    print( "Exact   SIG: 1.0000000000" )
    print( "Optimal SIG: %1.10f" % SIG_new )

    SIG_LIST = np.array( SIG_LIST )
    psi_T_LIST = np.array( psi_T_LIST )
    E_LIST     = np.array( E_LIST )


    # Plot the wavefunction as a function of iteration
    cmap = colormaps['brg'].resampled(256)
    cmap = cmap( np.linspace(0,1,len(SIG_LIST)) )
    plt.plot( xGRID, get_Vx(xGRID), c="black", lw=4, alpha=0.5, label="V(x)" )
    plt.plot( xGRID, psi_T_LIST[0], lw=1, c=cmap[0], alpha=0.5, label="$\\sigma_\\mathrm{Guess}$ = %1.2f" % SIG_LIST[0] )
    for i in range(1, len(SIG_LIST)-1):
        plt.plot( xGRID, psi_T_LIST[i], lw=1, c=cmap[i] )
    plt.plot( xGRID, psi_T_LIST[-1], lw=2, c="black", label="$\\sigma_\\mathrm{Final}$ = %1.2f" % SIG_LIST[-1] )
    plt.plot( xGRID, np.sign(np.sum(U_EXACT[:,0])) * U_EXACT[:,0] / np.max(np.abs(U_EXACT[:,0])), "--", lw=2, c="red", label="$\\sigma_\\mathrm{Exact}$ = %1.2f" % SIG_LIST[-1] )
    plt.legend( loc='upper right' )
    plt.xlabel("Position (a.u.)", fontsize=15)
    plt.ylabel("Energy / Wavefunction", fontsize=15)
    plt.ylim(0,2)
    plt.savefig("%s/psi.jpg" % DATA_DIR, dpi=300)
    plt.clf()

    plt.plot( np.arange(len(SIG_LIST)), SIG_LIST, lw=1 )
    plt.plot( np.arange(len(SIG_LIST)), SIG_LIST*0 + 1.0000000000, "--", lw=1, label="EXACT" )
    plt.legend( loc='upper right' )
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("$\\sigma$ (a.u.)", fontsize=15)
    plt.savefig("%s/SIG_convergence.jpg" % DATA_DIR, dpi=300)
    plt.clf()

    plt.loglog( np.arange(len(SIG_LIST)), np.abs(SIG_LIST - 1.0000000000), lw=1 )
    plt.ylim(1e-5,abs(SIG_LIST[0]))
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("$\\sigma$ (a.u.)", fontsize=15)
    plt.savefig("%s/SIG_error.jpg" % DATA_DIR, dpi=300)
    plt.clf()

if ( __name__ == "__main__" ):
    main()

