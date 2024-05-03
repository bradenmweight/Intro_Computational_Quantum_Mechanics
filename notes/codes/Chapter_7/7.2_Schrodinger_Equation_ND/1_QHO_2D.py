import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

DATA_DIR = "1_QHO_2D"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)


def get_Globals():
    global Nx, dx, xGRID, XMIN, XMAX
    Nx    = 25
    XMIN  = -5
    XMAX  =  5
    xGRID = np.linspace( XMIN, XMAX, Nx )
    dx    = xGRID[1] - xGRID[0]

    global Ny, dy, yGRID, YMIN, YMAX
    Ny    = 25
    YMIN  = -5
    YMAX  =  5
    yGRID = np.linspace( YMIN, YMAX, Ny )
    dy    = yGRID[1] - yGRID[0]

    global wx, wy
    wx = 1.0
    wy = 1.0

def get_Potential_Energy():

    def get_V( GRID, w ):
        return w * np.diag( GRID**2 ) / 2 

    Vx = get_V( xGRID, wx )
    Vy = get_V( yGRID, wy )

    V  = np.zeros( (Nx*Ny, Nx*Ny) )
    V += np.kron( Vx,         np.eye(Ny) )
    V += np.kron( np.eye(Nx), Vy         )
    return V

def get_Kinetic_Energy():

    def get_T( N, spacing ):
        T = np.zeros( (N, N) )
        for i in range(N):
            for j in range(N):
                if ( i == j ):
                    T[i,i] = np.pi**2 / 3
                if (i != j ):
                    T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
        return T / 2 / spacing**2

    Tx = get_T( Nx, dx )
    Ty = get_T( Ny, dy )

    T = np.zeros( (Nx*Ny, Nx*Ny) )
    T += np.kron( Tx,         np.eye(Ny) )
    T += np.kron( np.eye(Nx), Ty         )
    return T


def plot_Energies( E ):

    NPLOT    = 50
    indices  = np.arange( NPLOT )
    EXACT_1D = 1.0 + np.arange( NPLOT )
    plt.plot( indices, EXACT_1D[:NPLOT], "-", label="Exact 1D" )
    plt.plot( indices, E[:NPLOT], "o", label="NUM" )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/Energies.jpg", dpi=300 )
    plt.clf()

def plot_wavefunction( E, U ):

    for state in range( 10 ):
        plt.imshow( U[:,state].reshape( (Nx,Ny) ), cmap="afmhot_r", extent=[XMIN,XMAX,YMIN,YMAX] )
        plt.colorbar(pad=0.01)
        plt.savefig( f"{DATA_DIR}/WFN_{state}.jpg", dpi=300 )
        plt.clf()

def main():
    get_Globals()
    H          = get_Kinetic_Energy() + get_Potential_Energy()
    E, U       = np.linalg.eigh( H )
    print( "", np.round(E[:10],3) )
    plot_Energies( E )
    plot_wavefunction( E,  U )

if ( __name__ == "__main__" ):
    main()