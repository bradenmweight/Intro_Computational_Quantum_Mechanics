import numpy as np
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
import subprocess as sp
from time import time

DATA_DIR = "7.4.1_Schrodinger_Equation_Arbitrary_Basis/"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Params():
    global doANALYTIC_BASIS
    doANALYTIC_BASIS = False

    global xGRID, Nx, dx
    xMIN  = -10
    xMAX  =  10
    Nx    = 1001
    xGRID = np.linspace(xMIN, xMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    ### Define current system ###
    global Vx, WFN_0, E_0
    CASE = "DoubleWell" # "ISW" or "QHO" or "GAUSS" or "DoubleWell"


    if ( CASE == "QHO" ):
        # QHO
        w     = 0.15000
        Vx    = 0.5000 * w**2 * xGRID**2 
        WFN_0 = np.exp( -w * xGRID**2 / 2 )
        WFN_0 = WFN_0 / np.linalg.norm( WFN_0 )
        E_0   = 0.5000000000000000000000 * w

    elif ( CASE == "ISW" ):

        # Infinite square well
        L     = 4 # Width of the well
        Vx    = np.zeros( (Nx) )
        WFN_0 = np.cos( xGRID * np.pi / 2 )
        for xi,x in enumerate(xGRID):
            if ( x < -L/2 or x > L/2 ):
                Vx[xi] = 1000
                WFN_0[xi] = 0.0
            else:
                Vx[xi] = 0.0
                WFN_0[xi] = np.cos( x * np.pi / L )
        WFN_0 = WFN_0 / np.linalg.norm( WFN_0 )
        E_0   = np.pi**2 / 2 / L**2

    elif ( CASE == "GAUSS"  ):
        Vx    = -1 * np.exp( -xGRID**2 / 2 )
        Tx    = get_Tx( Nx, dx )
        E, U  = np.linalg.eigh( Tx + np.diag( Vx ) )
        E_0   = E[0]
        WFN_0 = U[:,0]

    elif ( CASE == "DoubleWell" ):
        Vx    = -1 * np.exp( -(xGRID+3)**2 / 2 ) - np.exp( -(xGRID-3)**2 / 2 )
        Tx    = get_Tx( Nx, dx )
        E, U  = np.linalg.eigh( Tx + np.diag( Vx ) )
        E_0   = E[0]
        WFN_0 = U[:,0]

    else:
        print(f"Potential chosen not found: {CASE}")
        exit()

    WFN_0 = np.sign( np.sum( WFN_0 ) ) * WFN_0 # Choose WFN to be positive

    #############################

def get_Potential_Matrix_Elements( Ubasis ):
    """
    Calculate the matrix elements of the potential operator.
    V_nm = <n|V|m> = \\int dx <n|x> V(x) <x|m> = \\int dx \\phi_n(x) V(x) \\phi_m(x)
    \\phi_n is the n_th QHO basis function
    #### THIS IS HARD TO DO ANALYTICALLY FOR ARBITRARY POTENTIALS ####
    #### FOR GAUSSIAN FUNCTIONS, THIS IS KNOWN FOR THE COULUMB POTENTIAL ####
    """
    V  = np.einsum("xn,x,xm->nm", Ubasis, Vx, Ubasis ) # nm matrix elements of \hat{V}
    return V

def get_Kinetic_Matrix_Elements( Ubasis ):
    """
    Calculate the matrix elements of the kinetic operator.
    T_nm = <n|T|m> = \\int dx dx' <n|x> T(x,x') <x'|m> = \\int dx dx' \\phi_n(x) T(x,x') \\phi_m(x')
    \\phi_n is the n_th QHO basis function
    """
    T = np.einsum("xn,xy,ym->nm", Ubasis, get_Tx( Nx, dx ), Ubasis ) # nm matrix elements of \hat{T}
    return T

def get_Kinetic_Matrix_Elements_Analytic( Ubasis ):
    """
    Calculate the matrix elements of the kinetic operator in the QHO basis
    <n|p^2|m> = w/4 <n|(a.T - a)^2|m> 
              = w/4 * [ (2m+1)*(n==m) + sqrt{(m+1)(m+2)}*(n==m+2) + sqrt{(m)(m+1)}*(n==m-2) ]
    """
    N = len(Ubasis[0,:]) # How many basis functions are there ?
    T = np.zeros( (N, N) )
    for n in range( N ):
        T[n,n] = (2 * n + 1)
        if ( n >= 2 ):
            m = n - 2
            T[n,m] = np.sqrt( (m+1) * (m+2) )
            T[m,n] = T[n,m] # Kinetic energy operator must be Hermitian
    return T * wbasis / 4

def get_FULL_H( Ubasis ):
    print("Calculating V matrix elements...fast since \\hat{V} is diagonal")
    V = get_Potential_Matrix_Elements( Ubasis )
    if ( doANALYTIC_BASIS == False ):
        print("Calculating T matrix elements...slow since \\hat{T} is not diagonal")
        T0 = time()
        T  = get_Kinetic_Matrix_Elements( Ubasis )
        print("Time to calculate T matrix elements: %1.4f s" % (time() - T0))
    else:
        print("Calculating T matrix elements...fast since \\hat{T} is analytic")
        T0 = time()
        T = get_Kinetic_Matrix_Elements_Analytic( Ubasis )
        print("Time to calculate T matrix elements: %1.4f s" % (time() - T0))
    return T + V

def do_Single_Point_Calculation():

    # First do single calculation with fixed basis size
    nbasis = 100
    print("Working on basis size: %d" % nbasis)
    Ubasis = get_Basis( nbasis )
    H      = get_FULL_H( Ubasis )
    E, U   = np.linalg.eigh( H )
    print( "Basis size: %d   Energy: %1.6f    Exact: %1.6f" % (nbasis, E[0], E_0) )
    plot_Single( E, U, Ubasis )

def do_Scan_Calculation():
    if ( doANALYTIC_BASIS == False ):
        print("\n\nERROR:")
        print("\tOnly do the scan calculation with analytic basis functions.")
        print("\tNumerical evaluation of the kinetic energy operator is too slow.\n")
        exit()
    # Do a scan over the number of basis functions
    nbasis_list = np.arange( 1,1000,5 )
    E_GS        = np.zeros( len(nbasis_list) )
    Ubasis      = get_Basis( nbasis_list[-1] )
    H           = get_FULL_H( Ubasis )
    for bi,b in enumerate(nbasis_list):
        H_TMP   = H[:b,:b]
        E, U    = np.linalg.eigh( H_TMP )
        print( "Basis size: %d   Energy: %1.6f    Exact: %1.8f" % (b, E[0], E_0) )
        E_GS[bi] = E[0]
    plot_scan( E_GS, nbasis_list )


def main():
    get_Params()

    # Do single point calculation
    do_Single_Point_Calculation()

    # Do scan over basis size
    do_Scan_Calculation()












#### I PUT EXTRA FUNCTION DOWN HERE -- NOT THE POINT OF THE SCRIPT ####


def plot_Single( E, U, Ubasis ):
    Ux    = Ubasis @ U[:,0] # Rotate to position basis from QHO basis
    Ux    = np.sign( np.sum( Ux ) ) * Ux # Choose WFN to be positive
    for i in range( len(Ubasis[0,:]) ):
        plt.plot(xGRID, np.abs(U[i,0]) * Ubasis[:,i], lw=1, alpha=0.5) # Scaled QHO basis function
    plt.plot(xGRID, WFN_0, c="black", lw=8, alpha=0.25, label="Exact GS")
    plt.plot(xGRID, Ux, c="red", lw=2, label="Numerical GS")
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Energy / Wavefunction", fontsize=15)
    plt.title("$E_0^\\mathrm{Num.}$ = %1.3f   $E_0^\\mathrm{Exact}$ = %1.3f    NBASIS = %d" % (E[0], E_0, len(Ubasis[0,:])), fontsize=15)
    plt.legend()
    #plt.xlim(-4,4)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/Ground_State_WFN.jpg", dpi=300)
    plt.clf()

    plt.plot(xGRID, Vx, c="black", lw=4, label="Potential")
    plt.plot(xGRID, WFN_0 + E_0, c="black", lw=8, alpha=0.25, label="Exact GS")
    plt.plot(xGRID, Ux + E[0], "--", c="red", lw=2, label="Numerical GS")
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Potential Energy", fontsize=15)
    plt.title("$E_0^\\mathrm{Num.}$ = %1.3f   $E_0^\\mathrm{Exact}$ = %1.3f    NBASIS = %d" % (E[0], E_0, len(Ubasis[0,:])), fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/Vx.jpg", dpi=300)
    plt.clf()

def plot_scan( E_GS, nbasis_list ):
    plt.semilogx(nbasis_list, nbasis_list*0 + E_0, "--", c="black", lw=3, label="Numerical")
    #plt.plot(nbasis_list, nbasis_list*0 + E_0, "--", c="black", lw=3, label="Numerical")
    #plt.plot(nbasis_list, E_GS, c="red", lw=3, label="Exact")
    plt.semilogx(nbasis_list, E_GS, c="red", lw=3, label="Exact")
    plt.legend()
    plt.xlabel("Number of Basis Functions", fontsize=15)
    plt.ylabel("Ground State Energy", fontsize=15)
    plt.title("Convergence of Ground State Energy", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Ground_State_Convergence.jpg", dpi=300)
    plt.clf()

    plt.loglog(nbasis_list, np.abs(E_GS - E_0), c="black", lw=3)
    plt.xlabel("Number of Basis Functions", fontsize=15)
    plt.ylabel("Ground State Energy", fontsize=15)
    plt.title("Convergence of Ground State Energy", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Ground_State_Convergence_Error.jpg", dpi=300)
    plt.clf()

def get_Tx( N, d ):
    """
    Construct the kinetic energy operator.
    """
    T  = np.zeros((N, N))
    T -= np.diag( np.ones(N-1), k=-1 )
    T += np.diag( 2*np.ones(N), k=0 )
    T -= np.diag( np.ones(N-1), k=1 )
    return T / 2 / d**2

def plot_Basis( V, E, U ):
    """
    Plot the QHO basis functions.
    """
    SIGN = np.sign( np.sum( U[:,0] ) ) # Choose WFN_0 to be positive
    NPLOT = 50
    plt.plot(xGRID, np.diagonal(V), c='black', lw=2, alpha=0.5)
    for n in range( NPLOT ):
        plt.plot(xGRID, 10 * SIGN * U[:,n] + E[n], lw=1, alpha=0.5)
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Wavefunction", fontsize=15)
    plt.title("Quantum Harmonic Oscillator Basis Functions", fontsize=15)
    plt.ylim( 0, E[NPLOT+10] )
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/QHO_Basis.jpg", dpi=300)
    plt.clf()

def get_Basis( NBASIS ):
    """
    Choose quantum harmonic oscillator basis functions.
    These are usually analytic, but we can also use numerical.
    Could be Gaussians, Slater-type orbitals, etc.
    """
    global wbasis
    wbasis = 5.0000
    V = 0.5000 * wbasis**2 * np.diag( xGRID**2 )
    T = get_Tx( Nx, dx )
    E, U = np.linalg.eigh( T + V )
    plot_Basis( V,E,U )
    return U[:,:NBASIS]

if ( __name__ == "__main__" ):
    main()