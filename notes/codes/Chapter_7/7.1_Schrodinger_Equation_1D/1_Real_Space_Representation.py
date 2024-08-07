import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

DATA_DIR = "1_Real_Space_Representation"
sp.call(f"mkdir -p DATA_DIR", shell=True)


def get_Globals():
    global Nx, dx, xGRID, XMIN, XMAX
    Nx    = 1001
    XMIN  = -10
    XMAX  =  10
    xGRID = np.linspace( XMIN, XMAX, Nx )
    dx    = xGRID[1] - xGRID[0]

def get_Potential_Energy():
    #V = np.zeros( (N,N) )
    # for i in range( N ):
    #     x = xGRID[i]
    #     V[i,i] = 0.5 * x**2

    V = np.diag( 0.5 * xGRID**2 )
    return V

def get_Kinetic_Energy():
    T = np.zeros( (Nx, Nx) )
    for i in range(Nx):
        for j in range(Nx):
            if ( i == j ):
                T[i,i] = 2
            if ( abs(i-j) == 1 ):
                T[i,j] = -1
    return T / 2 / dx**2

def get_Kinetic_Energy_DVR():
    T = np.zeros( (Nx, Nx) )
    for i in range(Nx):
        for j in range(Nx):
            if ( i == j ):
                T[i,i] = np.pi**2 / 3
            if (i != j ): # if ( not (i == j) ):
                T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
    return T / 2 / dx**2

def plot_Energies( E, EDVR ):

    NPLOT   = 50
    indices = np.arange( NPLOT )
    EXACT = 0.5 + np.arange( NPLOT )
    plt.plot( indices, EXACT[:NPLOT], "-", label="Exact" )
    plt.plot( indices, E[:NPLOT], "o", label="O1" )
    plt.plot( indices, EDVR[:NPLOT], "--", label="DVR" )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/Energies.jpg", dpi=300 )
    plt.clf()


    plt.plot( indices, EXACT - E[:NPLOT], "-", label="O1" )
    plt.plot( indices, EXACT - EDVR[:NPLOT], "-", label="DVR" )
    plt.legend()

    plt.savefig( f"{DATA_DIR}/Energies_DIFF.jpg", dpi=300 )
    plt.clf()

def plot_wavefunction( E, U ):

    plt.plot( xGRID, 0.5 * xGRID**2, "-", lw=6, c='black', label="V(x)" )
    plt.plot( xGRID, 3*U[:,0] + E[0], "-", label="$|E_0\\rangle$" )
    plt.plot( xGRID, 3*U[:,1] + E[1], "-", label="$|E_1\\rangle$" )
    plt.plot( xGRID, 3*U[:,2] + E[2], "-", label="$|E_2\\rangle$" )
    plt.xlim(-4,4)
    plt.ylim(0,4)
    plt.legend()

    plt.savefig( f"{DATA_DIR}/WFN.jpg", dpi=300 )
    plt.clf()

def main():
    get_Globals()
    H          = get_Kinetic_Energy() + get_Potential_Energy()
    E, U       = np.linalg.eigh( H )
    H_DVR      = get_Kinetic_Energy_DVR() + get_Potential_Energy()
    EDVR, UDVR = np.linalg.eigh( H_DVR )
    print( "", E[:10] )
    print( "", EDVR[:10] )
    plot_Energies( E, EDVR )
    plot_wavefunction( E,  U )

if ( __name__ == "__main__" ):
    main()