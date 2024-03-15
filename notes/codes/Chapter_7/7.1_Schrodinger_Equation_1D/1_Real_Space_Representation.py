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
            if (i != j ):
                T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
    return T / 2 / dx**2

def plot_Energies( E, EDVR ):

    EXACT = 0.5 + np.arange( len(E) )
    plt.plot( np.arange(len(E)), EXACT, "-", label="Exact" )
    plt.plot( np.arange(len(E)), E, "o", label="O1" )
    plt.plot( np.arange(len(E)), EDVR, "--", label="DVR" )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/Energies.jpg", dpi=300 )
    plt.clf()


    plt.plot( np.arange(len(E)), EXACT - E, "-", label="O1" )
    plt.plot( np.arange(len(E)), EXACT - EDVR, "-", label="DVR" )
    plt.legend()

    plt.savefig( f"{DATA_DIR}/Energies_DIFF.jpg", dpi=300 )
    plt.clf()


def main():
    get_Globals()
    H          = get_Kinetic_Energy() + get_Potential_Energy()
    E, U       = np.linalg.eigh( H )
    H_DVR      = get_Kinetic_Energy_DVR() + get_Potential_Energy()
    EDVR, UDVR = np.linalg.eigh( H_DVR )
    print( "", E[:20] )
    print( "", EDVR[:20] )
    plot_Energies( E, EDVR )
    #plot_wavefunction( U )

if ( __name__ == "__main__" ):
    main()