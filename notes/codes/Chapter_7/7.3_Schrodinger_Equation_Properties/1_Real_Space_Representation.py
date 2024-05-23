import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

DATA_DIR = "1_Real_Space_Representation"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)


def get_Globals():
    global Nx, dx, xGRID, XMIN, XMAX
    Nx    = 201
    XMIN  = -6
    XMAX  =  6
    xGRID = np.linspace( XMIN, XMAX, Nx )
    dx    = xGRID[1] - xGRID[0]

def get_Potential_Energy():
    #V = np.zeros( (N,N) )
    # for i in range( N ):
    #     x = xGRID[i]
    #     V[i,i] = 0.5 * x**2

    V = np.diag( xGRID**2 ) / 2 
    #V = -50*np.diag( xGRID**2 ) + 95*np.diag( xGRID**4 )
    return V - np.min( np.diagonal(V) ) * np.eye(Nx)

def get_Kinetic_Energy():
    T = np.zeros( (Nx, Nx) )
    for i in range(Nx):
        for j in range(Nx):
            if ( i == j ):
                T[i,i] = np.pi**2 / 3
            if (i != j ): # if ( not (i == j) ):
                T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
    return T / 2 / dx**2

def plot_Energies( E ):

    NPLOT   = 50
    indices = np.arange( NPLOT )
    plt.plot( indices, E[:NPLOT], "o" )
    plt.savefig( f"{DATA_DIR}/Energies.png", dpi=300 )
    plt.clf()

def plot_wavefunction( E, U ):

    V = get_Potential_Energy()
    V = np.diagonal( V )
    plt.plot( xGRID, V, "-", lw=6, c='black', label="V(x)" )
    for state in range( 5 ):
        plt.plot( xGRID, (E[5]-E[0])*U[:,state] + E[state], "-", label=f"$|E_{state}\\rangle$" )
    plt.xlim(XMIN,XMAX)
    plt.ylim(np.min(V),E[5]-E[0])
    plt.legend()
    plt.savefig( f"{DATA_DIR}/WFN.png", dpi=300 )
    plt.clf()

def plot_dipole_matrix( U ):
    # MU = np.zeros( (Nx, Nx) )
    # for i in range( Nx ):
    #     for j in range( Nx ):
    #         MU[i,j] = np.sum( U[:,i] * xGRID[:] * U[:,j] )
    MU  = np.einsum("xJ,x,xK->JK", U[:,:], xGRID[:], U[:,:])

    plt.imshow( MU[:10,:10], cmap="bwr", origin="lower" )
    plt.colorbar(pad=0.01)
    plt.xlabel("Electronic State, $j$",fontsize=15)
    plt.ylabel("Electronic State, $k$",fontsize=15)
    plt.savefig( f"{DATA_DIR}/Dipole.png", dpi=300 )
    plt.clf()

    return MU

def plot_momentum_matrix( U ):

    def get_P_OP_2Order():
        P_OP   = np.diag( np.ones(Nx-1),k=1 ) \
               - np.diag( np.ones(Nx-1),k=-1 )
        return -1j * P_OP / ( 2 * dx )
    def get_P_OP_4Order():
        P_OP  = -1*np.diag( np.ones(Nx-2),k= 2 ) \
                +8*np.diag( np.ones(Nx-1),k= 1 ) \
                -8*np.diag( np.ones(Nx-1),k=-1 ) \
                +1*np.diag( np.ones(Nx-2),k=-2 )
        return -1j * P_OP / ( 12 * dx )
    P_OP = get_P_OP_2Order()
    #P_OP = get_P_OP_4Order()

    # P    = np.zeros( (Nx, Nx), dtype=np.complex64 )
    # for i in range( 100 ):
    #     for j in range( 100 ):
    #         P[i,j] = U[:,i] @ P_OP[:,:] @ U[:,j]
    P = np.einsum("aJ,ab,bK->JK", U[:,:], P_OP[:,:], U[:,:])

    plt.imshow( np.imag(P[:10,:10]), cmap="bwr", origin="lower" )
    plt.colorbar(pad=0.01)
    plt.xlabel("Electronic State, $j$",fontsize=15)
    plt.ylabel("Electronic State, $k$",fontsize=15)
    plt.savefig( f"{DATA_DIR}/Momentum.png", dpi=300 )
    plt.clf()

    return P

def do_Phase_Correction( U ):
    # Choose ground state to be positive
    if ( np.sum(U[:,0]) < 0 ):
        print("Doing Phase Correction")
        U = -1 * U
    return U

def show_X_P_Relation( E, X_MAT, P_MAT ):
    X_from_P = np.zeros( (100,100) )
    for J in range( 100 ):
        for K in range( 100 ):
            if ( J != K ):
                X_from_P[J,K] = 1j * P_MAT[J,K] / (E[K] - E[J])
    
    print( "X:\n", np.round(X_MAT[:5,:5], 3) )
    print( "X from TRK:\n", np.round(X_from_P[:5,:5], 3) )
    print( "TRK - X:\n", X_from_P[:5,:5] - X_MAT[:5,:5] )

    P_from_X = np.zeros( (100,100), dtype=np.complex64 )
    for J in range( 100 ):
        for K in range( 100 ):
            if ( J != K ):
                P_from_X[J,K] = -1j * X_MAT[J,K] * (E[K] - E[J])

    print( "P:\n", np.round(P_MAT[:5,:5].imag, 3) )
    print( "P from TRK:\n", np.round(P_from_X[:5,:5].imag, 3) )
    print( "TRK - P:\n", P_from_X[:5,:5].imag - P_MAT[:5,:5].imag )

def show_TRK_Sum_Rule( E, X_MAT, P_MAT ):
    """
    Oscillator Strength:
    f_jk =  2 * |<j|X|k>|^2 * ( E_k - E_j )     (Length   Gauge)
         = -2 * |<j|P|k>|^2 / ( E_k - E_j )     (Velocity Gauge)
    \sum_{j!=k} f_{jk} = 1 (TRK Sum Rule)
    """
    f_X = np.zeros( (Nx,Nx) )
    f_P = np.zeros( (Nx,Nx) )
    for J in range( Nx ):
        for K in range( Nx ):
            if ( J != K ):
                    f_X[J,K] = 2 * np.abs( X_MAT[J,K] )**2 * ( E[J] - E[K] )
                    f_P[J,K] =  2 * np.abs( P_MAT[J,K] )**2 / ( E[J] - E[K] )

    N_TRUNC = 20 # Sum rule, in principle, is only satisfied when N_TRUNC -> \infty
    print("TRK Sum Rule:")
    print("NSTATES:", N_TRUNC)
    print("(X-Gauge) \sum_{j!=k} f_{jk}:", np.round(np.einsum( "JK->K", f_X[:N_TRUNC,:10] ) ,3) )
    print("(P-Gauge) \sum_{j!=k} f_{jk}:", np.round(np.einsum( "JK->K", f_P[:N_TRUNC,:10] ) ,3) )
    print("Accuracy of P-Gauge depends on the accuracy of the finite difference derivative.")

def plot_Absorption_Spectra( E, X_MAT, P_MAT ):

    dE_0K = E[:] - E[0]

    # Compute ground-to-excited oscillator strength
    f_X = np.zeros( (Nx) )
    f_P = np.zeros( (Nx) )
    for J in range( 1, Nx ):
        f_X[J] = 2 * np.abs( X_MAT[J,0] )**2 * dE_0K[J]
        f_P[J] = 2 * np.abs( P_MAT[J,0] )**2 / dE_0K[J]

    # Compute absorption spectra
    SIG   = 0.1
    EMIN  = 0
    EMAX  = dE_0K[5]
    NPTS  = 1000
    EGRID = np.linspace( EMIN, EMAX, NPTS )

    ABS_X = np.zeros( NPTS )
    ABS_P = np.zeros( NPTS )
    for pt in range( NPTS ):
        ABS_X[pt] = np.sum( f_X[1:] * np.exp( -(EGRID[pt]-dE_0K[1:])**2/2/SIG**2 ) )
        ABS_P[pt] = np.sum( f_P[1:] * np.exp( -(EGRID[pt]-dE_0K[1:])**2/2/SIG**2 ) )

    plt.plot( EGRID, ABS_X, "-" , lw=4, c="black", label="Length Gauge" )
    plt.plot( EGRID, ABS_P, "--", lw=2, c="red"  , label="Velocity Gauge" )
    markerline, stemlines, baseline = plt.stem(dE_0K[1:], f_X[1:], linefmt="blue", markerfmt="o" )
    markerline.set_markerfacecolor('none')
    markerline.set_markeredgecolor('blue')
    markerline.set_markersize(8)
    markerline.set_markeredgewidth(1.5)
    markerline, stemlines, baseline = plt.stem(dE_0K[1:], f_P[1:], linefmt="red", markerfmt="." )
    markerline.set_markerfacecolor('none')
    markerline.set_markeredgecolor('red')
    markerline.set_markersize(4)
    markerline.set_markeredgewidth(0.75)
    plt.xlim(EMIN,EMAX)
    plt.xlabel("Energy, $E$", fontsize=15)
    plt.ylabel("Absorption Spectra", fontsize=15)
    plt.legend()
    plt.savefig( f"{DATA_DIR}/Absorption.png", dpi=300 )



def main():
    get_Globals()
    H          = get_Kinetic_Energy() + get_Potential_Energy()
    E, U       = np.linalg.eigh( H )
    print( "", E[:10] )
    plot_Energies( E )
    U = do_Phase_Correction( U )
    plot_wavefunction( E,  U )
    X_MAT  = plot_dipole_matrix( U )
    P_MAT  = plot_momentum_matrix( U )
    show_X_P_Relation( E, X_MAT, P_MAT )
    show_TRK_Sum_Rule( E, X_MAT, P_MAT )
    plot_Absorption_Spectra( E, X_MAT, P_MAT )

if ( __name__ == "__main__" ):
    main()