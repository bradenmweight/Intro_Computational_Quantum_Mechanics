import numpy as np
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
import subprocess as sp
from numba import njit
from time import time

DATA_DIR = "7.4.2_Schrodinger_Equation_Vibrational_Polaritons/"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Params():
    global xGRID, Nx, dx
    xMIN  = -5
    xMAX  =  5
    Nx    =  25
    xGRID = np.linspace(xMIN, xMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    # FOR FOCK BASIS
    global Nfock, a_op, adag_op
    Nfock   = 20 # |0>, |1>, ..., |Nfock-1> -- Photon Basis Size
    a_op    = np.diag( np.sqrt( np.arange(1,Nfock) ), k=1 )
    adag_op = a_op.T

    # FOR GRID BASIS
    global qcGRID, Nqc, dqc
    qcMIN  = -5
    qcMAX  =  5
    Nqc    =  25
    qcGRID = np.linspace(qcMIN, qcMAX, Nqc)
    dqc    = qcGRID[1] - qcGRID[0]

    ### Define current molecular system ###
    CASE = "QHO" # "ISW" or "QHO" or "GAUSS" or "DoubleWell"
    get_Molecular_System( CASE )

@njit
def get_Tx( N, d ):
    """
    Construct the kinetic energy operator.
    """
    T  = np.zeros((N, N))
    for xi in range( N ):
        for xj in range( N ):
            if ( xi == xj ):
                T[xi, xj] = np.pi**2 / 3
            else:
                T[xi, xj] = 2 * (-1)**(xi - xj) / (xi - xj)**2
    return T / 2 / d**2

@njit
def get_H_Jaynes_Cummings_FOCK( wc = 1.0, A0 = 0.01 ):
    H = np.zeros( (Nx*Nfock, Nx*Nfock) )

    I_M = np.eye(Nx)
    I_F = np.eye(Nfock)

    H_M   = get_Tx( Nx, dx ) + np.diag( Vx )
    H_PH  = wc * a_op.T @ a_op
    H_INT = wc * A0 * np.kron(np.diag(xGRID), adag_op + a_op) # Choose dipole operator to be x
    H_DSE = wc * A0**2 * np.kron( np.diag(xGRID**2), I_F ) # Choose dipole squared operator to be x squared

    H += np.kron( H_M, I_F )
    H += np.kron( I_M, H_PH )
    H += H_INT + H_DSE

    return H

@njit
def get_H_Jaynes_Cummings_qc( wc = 1.0, A0 = 0.01 ):
    H = np.zeros( (Nx*Nqc, Nx*Nqc) )

    I_M = np.eye(Nx)
    I_F = np.eye(Nqc)

    H_M   = get_Tx( Nx, dx ) + np.diag( Vx )
    H_PH  = get_Tx( Nqc, dqc ) + np.diag( 0.5*wc**2*qcGRID**2 )
    H_INT = np.sqrt(2 * wc**3) * A0 * np.kron(np.diag(xGRID), np.diag(qcGRID)) # Choose dipole operator to be x
    H_DSE = wc * A0**2 * np.kron( np.diag(xGRID**2), I_F ) # Choose dipole operator to be x

    H += np.kron( H_M, I_F )
    H += np.kron( I_M, H_PH )
    H += H_INT + H_DSE

    return H

def do_Single_Point_Calculation( wc = 1.0, A0 = 0.01 ):

    # First do single calculation with fixed basis size
    Tfock         = time()
    Efock, Ufock  = np.linalg.eigh( get_H_Jaynes_Cummings_FOCK( wc=wc, A0=A0 ) )
    Tfock         = time() - Tfock
    Tqc           = time()
    Eqc,   Uqc    = np.linalg.eigh( get_H_Jaynes_Cummings_qc( wc=wc, A0=A0 ) )
    Tqc           = time() - Tqc
    return Efock, Eqc, [Tfock, Tqc]

def do_wc_scan_Calculation():
    A0      = 0.05 # Coupling Strength
    wc_list = np.linspace(0.5, 1.5, 31) # Cavity Frequency
    Efock   = np.zeros( (len(wc_list), Nx*Nfock) )
    Eqc     = np.zeros( (len(wc_list), Nx*Nqc) )
    for wci,wc in enumerate(wc_list):
        print( wci, wc )
        Efock[wci,:], Eqc[wci,:], times = do_Single_Point_Calculation( wc=wc, A0=A0 )
        print( "  Fock Basis (%d) = %1.3f s    qc Basis (%d) = %1.3f s" % (Nfock, times[0], Nqc, times[1]) )
    return wc_list, Efock, Eqc, A0

def do_A0_scan_Calculation():
    wc      = 1.0 # Cavity Frequency
    A0_list = np.linspace(0.0, 1.0, 21) # Coupling Strength
    Efock   = np.zeros( (len(A0_list), Nx*Nfock) )
    Eqc     = np.zeros( (len(A0_list), Nx*Nqc) )
    for A0i,A0 in enumerate(A0_list):
        print( A0i, wc )
        Efock[A0i,:], Eqc[A0i,:], times = do_Single_Point_Calculation( wc=wc, A0=A0 )
        print( "  Fock Basis (%d) = %1.3f s    qc Basis (%d) = %1.3f s" % (Nfock, times[0], Nqc, times[1]) )
    return A0_list, Efock, Eqc, wc

def plot_scan( the_list, Efock, Eqc, title, EREF, fixed_value ):
    for state in range( Efock.shape[1] ):
        if ( state == 0 ):
            plt.plot( the_list, Efock[:,state] - EREF[0], "-", lw=5, c='black', alpha=0.5, label=f"Fock" )
            plt.plot( the_list, Eqc[:,state] - EREF[1], "-o", lw=3, mfc='none', c="blue", label=f"qc" )
        else:
            plt.plot( the_list, Efock[:,state] - EREF[0], "-", lw=5, c='black', alpha=0.5 )
            plt.plot( the_list, Eqc[:,state] - EREF[1], "-o", lw=2, mfc='none', c="blue" )
    if ( title == "A0" ):
        plt.ylabel( "Absolute Energy, $E_n(A_0) - E_0(A_0 = 0.0)$ (a.u.)", fontsize=15 )
        plt.title( "$\\omega_\\mathrm{c}$ = %1.2f a.u." % (fixed_value), fontsize=15 )
        plt.xlabel( "Coupling Strength, $A_0$ (a.u.)", fontsize=15 )
    else:
        plt.ylabel( "Transition Energy, $E_n(\\omega_\\mathrm{c}) - E_0(\\omega_\\mathrm{c})$ (a.u.)", fontsize=15 )
        plt.title( "$A_0$ = %1.2f a.u." % (fixed_value), fontsize=15 )
        plt.xlabel( "Cavity Frequency, $\\omega_\\mathrm{c}$ (a.u.)", fontsize=15 )
    plt.legend()
    plt.xlim(the_list[0],the_list[-1])
    plt.ylim(0,4.0)
    plt.savefig( f"{DATA_DIR}/{title}_scan.jpg", dpi=300 )
    plt.clf()


def main():
    get_Params()

    # Do single point calculation
    Efock, Eqc, times = do_Single_Point_Calculation()
    print( "Fock Basis", Efock[:4] - Efock[0] ) 
    print( "qc Basis", Eqc[:4] - Eqc[0] )
    print( "  Fock Basis (%d) = %1.3f s    qc Basis (%d) = %1.3f s" % (Nfock, times[0], Nqc, times[1]) )

    # Scan over the cavity frequency
    wc_list, Efock, Eqc, A0 = do_wc_scan_Calculation()
    plot_scan( wc_list, Efock, Eqc, title="wc", EREF=[Efock[:,0],Eqc[:,0]], fixed_value=A0 )

    # Scan over the coupling strength
    A0_list, Efock, Eqc, wc = do_A0_scan_Calculation()
    plot_scan( A0_list, Efock, Eqc, title="A0", EREF=[Efock[0,0],Eqc[0,0]], fixed_value=wc )

    ##########################
    # TODO -- HOMEWORK -- TODO
    # Write a code to scan over the number of basis states and plot the error
    #    with the NFOCK = 100 and Nx = 100 as the reference energy
    # TODO -- HOMEWORK -- TODO
    ##########################

    ##########################
    # TODO -- HOMEWORK -- TODO
    # Modify the plots to show the average photon number <N> = <a.T a> for each data point
    #    and from both photon basis states (start with Fock -- easier)
    # TODO -- HOMEWORK -- TODO
    ##########################

    ##########################
    # TODO -- HOMEWORK -- TODO
    # Plot the wavefunction of the ground state for various values of coupling strength A0
    # Plot the wavefunction of the upper and lower polaritonic states at resonance
    #   for zero coupling A0 = 0.0 a.u. and at finite coupling A0 = 0.05 a.u.
    # For plotting the wavefunctions, use the qc grid basis 
    # TODO -- HOMEWORK -- TODO
    ##########################
















def get_Molecular_System( CASE ):
    global Vx, WFN_0, E_0

    if ( CASE == "QHO" ):
        # QHO
        w     = 1.0000
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







if ( __name__ == "__main__" ):
    main()