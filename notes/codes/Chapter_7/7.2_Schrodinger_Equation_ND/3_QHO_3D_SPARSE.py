import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
from time import time
from scipy.sparse import dok_matrix
from scipy.sparse import kron as sparse_kron
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import eigsh as sparse_eigh

DATA_DIR = "3_QHO_3D_SPARSE"
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

    global Ixsparse, Iysparse, Izsparse
    Ixsparse = sparse_identity( Nx )
    Iysparse = sparse_identity( Ny )
    Izsparse = sparse_identity( Nz )

def get_H_dense():

    def get_V():

        Vx = wx * np.diag( xGRID**2 ) / 2 
        Vy = wy * np.diag( yGRID**2 ) / 2 
        Vz = wz * np.diag( zGRID**2 ) / 2 

        V  = np.zeros( (Nx*Ny*Nz, Nx*Ny*Nz) )
        V += np.kron( Vx, np.kron( Iy, Iz ) )
        V += np.kron( Ix, np.kron( Vy, Iz ) )
        V += np.kron( Ix, np.kron( Iy, Vz ) )
        return V

    def get_T():

        def get_T_MAT( N, spacing ):
            T  = np.zeros( (N, N) )
            T -= np.diag( np.ones(N-1), k=-1 )
            T += np.diag( np.ones(N)*2, k=0 )
            T -= np.diag( np.ones(N-1), k=1 )
            return T / 2 / spacing**2

        Tx = get_T_MAT( Nx, dx )
        Ty = get_T_MAT( Ny, dy )
        Tz = get_T_MAT( Nz, dz )

        T  = np.zeros( (Nx*Ny*Nz, Nx*Ny*Nz) )
        T += np.kron( Tx, np.kron( Iy, Iz ) )
        T += np.kron( Ix, np.kron( Ty, Iz ) )
        T += np.kron( Ix, np.kron( Iy, Tz ) )

        return T

    H = get_T() + get_V()
    print( "Dense:  Nx Ny Nz = %d %d %d\tdim(H) = %1.0f\tMemory  = %1.3f GB" % (Nx,Ny,Nz,len(H),H.data.nbytes * 1e-9) )
    E, _ = np.linalg.eigh( H )
    return E

def get_H_sparse():

    def get_V():

        Vx = wx * np.diag( xGRID**2 ) / 2 
        Vy = wy * np.diag( yGRID**2 ) / 2 
        Vz = wz * np.diag( zGRID**2 ) / 2 

        V  = dok_matrix( (Nx*Ny*Nz, Nx*Ny*Nz) )
        V += sparse_kron( Vx, sparse_kron( Iysparse, Izsparse ) )
        V += sparse_kron( Ixsparse, sparse_kron( Vy, Izsparse ) )
        V += sparse_kron( Ixsparse, sparse_kron( Iysparse, Vz ) )
        return V

    def get_T():

        def get_T_MAT( N, spacing ):
            T  = np.zeros( (N, N) )
            T -= np.diag( np.ones(N-1), k=-1 )
            T += np.diag( np.ones(N)*2, k=0 )
            T -= np.diag( np.ones(N-1), k=1 )
            return T / 2 / spacing**2

        Tx = get_T_MAT( Nx, dx )
        Ty = get_T_MAT( Ny, dy )
        Tz = get_T_MAT( Nz, dz )

        T  = dok_matrix( (Nx*Ny*Nz, Nx*Ny*Nz) )
        T += sparse_kron( Tx, sparse_kron( Iysparse, Izsparse ) )
        T += sparse_kron( Ixsparse, sparse_kron( Ty, Izsparse ) )
        T += sparse_kron( Ixsparse, sparse_kron( Iysparse, Tz ) )
        return T

    H = get_T() + get_V()
    print( "Sparse: Nx Ny Nz = %d %d %d\tdim(H) = %1.0f\tMemory  = %1.3f GB   %1.3f MB" % (Nx,Ny,Nz,H.shape[0],H.data.nbytes * 1e-9, H.data.nbytes * 1e-6) )
    NROOTS = 1 # How many eigenvalues to compute ?
    E, _ = sparse_eigh( H, k=NROOTS, which="SA", return_eigenvectors=True )
    return E

def plot( n_list, TIMES_dense, TIMES_sparse, E0_dense, E0_sparse):

    plt.plot( n_list, TIMES_dense, "o-", c="black", label="Dense" )
    plt.plot( n_list, TIMES_sparse, "o-", c="red", label="Sparse" )
    plt.xlabel("Number of Grid Points, $N = N_\mathrm{x} = N_\mathrm{y} = N_\mathrm{z}$", fontsize=15)
    plt.ylabel("Calculation Time (seconds)", fontsize=15)
    plt.title("Computational Scaling", fontsize=15)
    plt.xlim(n_list[0], n_list[-1])
    plt.tight_layout()
    plt.legend()
    plt.savefig( f"{DATA_DIR}/TIMES.jpg", dpi=300 )
    plt.clf()

    plt.plot( n_list, n_list*0 + 1.500000000, "-", c="black", label="Exact" )
    plt.plot( n_list, E0_dense, "-", lw=6, alpha=0.5, c="black", label="Dense" )
    plt.plot( n_list, E0_sparse, "o-", c="red", label="Sparse" )
    plt.xlabel("Number of Grid Points, $N = N_\mathrm{x} = N_\mathrm{y} = N_\mathrm{z}$", fontsize=15)
    plt.ylabel("Ground State Energy (a.u.)", fontsize=15)
    plt.xlim(n_list[0], n_list[-1])
    plt.tight_layout()
    plt.legend()
    plt.savefig( f"{DATA_DIR}/GS_ENERGY.jpg", dpi=300 )
    plt.clf()

    plt.semilogy( n_list, np.abs(1.500000000 - E0_dense), "-", lw=6, alpha=0.5, c="black", label="Dense" )
    plt.semilogy( n_list, np.abs(1.500000000 - E0_sparse), "o-", c="red", label="Sparse" )
    plt.xlabel("Number of Grid Points, $N = N_\mathrm{x} = N_\mathrm{y} = N_\mathrm{z}$", fontsize=15)
    plt.ylabel("Ground State Energy (a.u.)", fontsize=15)
    plt.xlim(n_list[0], n_list[-1])
    plt.tight_layout()
    plt.legend()
    plt.savefig( f"{DATA_DIR}/GS_ENERGY_ERROR.jpg", dpi=300 )
    plt.clf()

def main():

    n_list       = np.arange(5,10,1)
    n_list       = np.arange(10,100,10)
    n_list       = np.append( n_list, np.arange(100,600,100) )
    TIMES_dense  = np.zeros( len(n_list) )
    TIMES_sparse = np.zeros( len(n_list) )
    E0_dense     = np.zeros( len(n_list) )
    E0_sparse    = np.zeros( len(n_list) )

    for ni,n in enumerate( n_list ):
        get_Globals(n,n,n)

        #### DENSE version ####
        if ( n <= 16 ):
            T0              = time()
            E0_dense[ni]    = get_H_dense()[0]
            TIMES_dense[ni] = time() - T0
            print("Dense Error:  %1.6f 1.50000000000" % (E0_dense[ni]))

        else:
            TIMES_dense[ni] = np.nan
            E0_dense[ni]    = np.nan

        #### SPARSE version ####
        T0               = time()
        E0_sparse[ni]    = get_H_sparse()[0]
        TIMES_sparse[ni] = time() - T0
        print("Sparse Error: %1.6f 1.50000000000" % (E0_sparse[ni]))
    plot( n_list, TIMES_dense, TIMES_sparse, E0_dense, E0_sparse )



if ( __name__ == "__main__" ):
    main()