import numpy as np
import scipy as sc
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
import subprocess as sp
from time import time

DATA_DIR = "7.4.3_Schrodinger_Equation_NonOrthogonal_Basis/"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Params( shift1 ):
    global shift
    shift = shift1 # Shift the basis functions by this amount

    global CASE 
    CASE = "DoubleWell" # "ISW" or "QHO" or "GAUSS" or "DoubleWell"

    global do_Orthogonalization
    do_Orthogonalization = True # Do we orthogonalize the basis functions ?


def get_Current_Potential():

    global Vx, WFN_0, E_0
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

def get_FULL_H( Ubasis ):
    get_Current_Potential()
    print("Calculating V matrix elements...fast since \\hat{V} is diagonal")
    V = get_Potential_Matrix_Elements( Ubasis )
    print("Calculating T matrix elements...slow since \\hat{T} is not diagonal")
    T0 = time()
    T  = get_Kinetic_Matrix_Elements( Ubasis )
    print("Time to calculate T matrix elements: %1.4f s" % (time() - T0))
    return T + V

def do_Single_Point_Calculation():
    shift = [0.0]
    get_Params( shift1=shift )

    # First do single calculation with fixed basis size
    nbasis = 50
    print("Working on basis size: %d" % nbasis)
    Ubasis = get_Basis( nbasis // len(shift) ) # Only use NBASIS as total number of basis functions
    print("Basis Size: ", Ubasis.shape)
    H      = get_FULL_H( Ubasis )
    E, U   = np.linalg.eigh( H )
    print( "Basis size: %d   Energy: %1.6f    Exact: %1.6f" % (nbasis, E[0], E_0) )
    plot_Single( E, U, Ubasis )

def do_Scan_Calculation():

    shift_list  = [[0.0], [-3.0, 3.0]]
    nbasis_list = np.array([5, 10, 15, 20, 30, 40, 50])
    E_GS        = np.zeros( (len(nbasis_list), len(shift_list)) )

    for si,s in enumerate( shift_list ):
        get_Params( shift1=s )
        for bi,b in enumerate(nbasis_list):
            Ubasis  = get_Basis( b//len(s) ) # Only use NBASIS as total number of basis functions
            H       = get_FULL_H( Ubasis )
            print(s, bi, Ubasis.shape)
            E, U    = np.linalg.eigh( H )
            print( s, "Basis size: %d   Energy: %1.6f    Exact: %1.8f" % (b, E[0], E_0) )
            E_GS[bi,si] = E[0]
    plot_scan( E_GS, nbasis_list, shift_list )


def main():

    # Do single point calculation
    do_Single_Point_Calculation()

    # Do scan over basis size
    #do_Scan_Calculation()












#### I PUT EXTRA FUNCTION DOWN HERE -- NOT THE POINT OF THE SCRIPT ####


def plot_Single( E, U, Ubasis ):
    Ux    = Ubasis @ U[:,0] # Rotate to position basis from QHO basis
    Ux    = np.sign( np.sum( Ux ) ) * Ux # Choose WFN to be positive
    for i in range( len(Ubasis[0,:]) ):
        #plt.plot(xGRID, np.abs(U[i,0]) * Ubasis[:,i], lw=1, alpha=0.5) # Scaled QHO basis function
        plt.plot(xGRID, U[i,0] * Ubasis[:,i], lw=1, alpha=0.5) # Scaled QHO basis function
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

def plot_scan( E_GS, nbasis_list, shift_list ):
    plt.semilogx(nbasis_list, nbasis_list*0 + E_0, "--", c="black", lw=3, label="Numerical")
    #plt.plot(nbasis_list, nbasis_list*0 + E_0, "--", c="black", lw=3, label="Numerical")
    #plt.plot(nbasis_list, E_GS, c="red", lw=3, label="Exact")
    for si, s in enumerate(shift_list):
        plt.semilogx(nbasis_list, E_GS[:,si], "-o", lw=3, label=s)
    plt.legend()
    plt.xlabel("Number of Basis Functions", fontsize=15)
    plt.ylabel("GS Energy (a.u.)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Ground_State_Convergence.jpg", dpi=300)
    plt.clf()

    for si, s in enumerate(shift_list):
        plt.loglog(nbasis_list, np.abs(E_GS[:,si] - E_0), "-o", lw=3, label=s)
    plt.legend()
    plt.xlabel("Number of Basis Functions", fontsize=15)
    plt.ylabel("Error in GS Energy (a.u.)", fontsize=15)
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

def plot_Basis( wbasis, E, U_NO, U_OR, S_NO, S_OR ):
    """
    Plot the QHO basis functions.
    """
    # Correct phases of basis functions such that PSI_0 is positive
    NBASIS = len(E) // len(shift)
    phase  = np.zeros( len(shift) )
    for si in range( len(shift) ):
        ind = NBASIS * si
        phase[si] = np.sign( np.sum(U_NO[:,ind]) )
    for si,s in enumerate( shift ):
        plt.plot(xGRID, 0.5000 * wbasis**2 * (xGRID - s)**2, c='black', lw=2, alpha=0.5)
        ind = NBASIS * si 
        for n in range( len(U_NO[0])//len(shift) ):
            plt.plot(xGRID, 10 * phase[si] * U_NO[:,ind+n] + E[ind+n], lw=1, alpha=0.5)
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Wavefunction", fontsize=15)
    plt.title("Quantum Harmonic Oscillator Basis Functions", fontsize=15)
    plt.ylim( 0, 10 )
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/QHO_Basis_Nonorthogonal.jpg", dpi=300)
    plt.clf()


    # Correct phases of basis functions such that PSI_0 is positive
    NBASIS = len(E) // len(shift)
    phase  = np.zeros( len(shift) )
    for si in range( len(shift) ):
        ind = NBASIS * si
        phase[si] = np.sign( np.sum(U_OR[:,ind]) )
    for si,s in enumerate( shift ):
        plt.plot(xGRID, 0.5000 * wbasis**2 * (xGRID - s)**2, c='black', lw=2, alpha=0.5)
        ind = NBASIS * si 
        for n in range( len(U_OR[0])//len(shift) ):
            plt.plot(xGRID, 10 * phase[si] * U_OR[:,ind+n] + E[ind+n], lw=1, alpha=0.5)
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Wavefunction", fontsize=15)
    plt.title("Quantum Harmonic Oscillator Basis Functions", fontsize=15)
    plt.ylim( 0, 10 )
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/QHO_Basis_Orthogonal.jpg", dpi=300)
    plt.clf()



    plt.imshow(S_NO, cmap="bwr")
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/Overlap_Matrix_Basis_Nonorthogonal.jpg", dpi=300)
    plt.clf()

    plt.imshow(S_OR, cmap="bwr")
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/Overlap_Matrix_Basis_Orthogonalized.jpg", dpi=300)
    plt.clf()

def lowdin_orthogonalization( U, S ):
    u,s,vt    = np.linalg.svd( S )
    Shalfinv  = u @ np.diag(np.sqrt(1/s)) @ vt
    U         = np.einsum("jk,xk->xj", Shalfinv, U)
    return U


def get_Basis( NBASIS ):
    """
    Choose quantum harmonic oscillator basis functions.
    These are usually analytic, but we can also use numerical.
    Could be Gaussians, Slater-type orbitals, etc.
    """
    global NBASIS_new
    NBASIS_new = NBASIS * len(shift)

    # Define grid to ensure that basis functions are well defined
    global xGRID, Nx, dx, xMIN, xMAX
    xMIN   = np.min(shift) - 10
    xMAX   = np.max(shift) + 10
    Nx     = 1001
    xGRID  = np.linspace(xMIN, xMAX, Nx)
    dx     = xGRID[1] - xGRID[0]

    global wbasis
    wbasis = 1.0000
    T      = get_Tx( Nx, dx )

    E_NO = []
    U_NO = []
    for s in shift:
        V      = 0.5000 * wbasis**2 * np.diag( (xGRID - s)**2 )
        e, u   = np.linalg.eigh( T + V )
        E_NO.append( e[:NBASIS] )
        U_NO.append( u[:,:NBASIS] )
    E_NO = np.array(E_NO).reshape( (NBASIS_new) )
    U_NO = np.array(U_NO).swapaxes(0,1).reshape( (Nx,NBASIS_new) )
    
    S_NO = np.einsum("xj,xk->jk", U_NO, U_NO) # No dx factor here, since 1/sqrt(dx) is implied
    U_OR = lowdin_orthogonalization( U_NO, S_NO )
    S_OR = np.einsum("xj,xk->jk", U_OR, U_OR) # No dx factor here, since 1/sqrt(dx) is implied

    plot_Basis( wbasis, E_NO, U_NO, U_OR, S_NO, S_OR )
    
    if ( do_Orthogonalization == True ):
        return U_OR
    else:
        return U_NO

if ( __name__ == "__main__" ):
    main()