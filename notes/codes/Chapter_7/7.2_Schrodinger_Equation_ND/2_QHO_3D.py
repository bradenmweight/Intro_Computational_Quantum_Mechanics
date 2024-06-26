import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
from time import time

DATA_DIR = "2_QHO_3D"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)


def get_Globals( nx,ny,nz ):
    global Nx, dx, xGRID, XMIN, XMAX
    Nx    = nx
    XMIN  = -5
    XMAX  =  5
    xGRID = np.linspace( XMIN, XMAX, Nx )
    dx    = xGRID[1] - xGRID[0]

    global Ny, dy, yGRID, YMIN, YMAX
    Ny    = ny
    YMIN  = -5
    YMAX  =  5
    yGRID = np.linspace( YMIN, YMAX, Ny )
    dy    = yGRID[1] - yGRID[0]

    global Nz, dz, zGRID, ZMIN, ZMAX
    Nz    = nz
    ZMIN  = -5
    ZMAX  =  5
    zGRID = np.linspace( ZMIN, ZMAX, Nz )
    dz    = zGRID[1] - zGRID[0]

    global wx, wy, wz
    wx = 1.0
    wy = 1.0
    wz = 1.0

    global Ix, Iy, Iz
    Ix = np.eye( Nx )
    Iy = np.eye( Ny )
    Iz = np.eye( Nz )

def get_Potential_Energy():

    def get_V( GRID, w ):
        return w * np.diag( GRID**2 ) / 2 

    Vx = get_V( xGRID, wx )
    Vy = get_V( yGRID, wy )
    Vz = get_V( zGRID, wz )

    V  = np.zeros( (Nx*Ny*Nz, Nx*Ny*Nz) )
    V += np.kron( Vx, np.kron( Iy, Iz ) )
    V += np.kron( Ix, np.kron( Vy, Iz ) )
    V += np.kron( Ix, np.kron( Iy, Vz ) )

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
    Tz = get_T( Nz, dz )

    T = np.zeros( (Nx*Ny*Nz, Nx*Ny*Nz) )
    T += np.kron( Tx, np.kron( Iy, Iz ) )
    T += np.kron( Ix, np.kron( Ty, Iz ) )
    T += np.kron( Ix, np.kron( Iy, Tz ) )

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

def plot_times( n_list, TIMES ):

    n_fine = np.linspace( n_list[0], n_list[-1], 100 )

    plt.plot( n_list, TIMES, "o-", c="black", label="Numerical" )
    plt.plot( n_fine, 0.0004 * 2**n_fine, "--", c="red", label="$\sim 2^N$" )
    plt.plot( n_fine, 0.001 * n_fine**3, "--", c="blue", label="$\sim N^3$" )
    plt.xlabel("Number of Grid Points, N", fontsize=15)
    plt.ylabel("Calculation Time (seconds)", fontsize=15)
    plt.title("Computational Scaling", fontsize=15)
    plt.xlim(n_list[0], n_list[-1])
    plt.tight_layout()
    plt.legend()
    plt.savefig( f"{DATA_DIR}/TIMES.jpg", dpi=300 )
    plt.clf()

def main():

    #n_list = np.arange( 3,18 )
    n_list = np.arange( 3,16 )
    TIMES  = np.zeros( len(n_list) )

    for ni,n in enumerate( n_list ):
        print("Working on Nx = ", n) 
        T0 = time()
        get_Globals(n,n,n)
        E, U       = np.linalg.eigh( get_Kinetic_Energy() + get_Potential_Energy() )
        TIMES[ni]  = time() - T0


        #print( "", np.round(E[:10],3) )
        #plot_Energies( E )
        #plot_wavefunction( E,  U )

    plot_times( n_list, TIMES )

if ( __name__ == "__main__" ):
    main()