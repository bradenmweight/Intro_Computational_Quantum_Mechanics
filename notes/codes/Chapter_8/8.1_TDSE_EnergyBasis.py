import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "8.1_TDSE_EnergyBasis"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global Nx, dx, xGRID
    XMIN  = -10.0
    XMAX  = 10.0
    Nx    = 1000
    xGRID = np.linspace(XMIN, XMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    global dt, tGRID
    tMAX  = 10
    dt    = 0.1
    tGRID = np.arange(0, tMAX+dt, dt)


def propagate( E, psi_0, name="" ):
    ### Propagate the wavefunction in time
    psi_t = np.zeros( (len(tGRID), len(E)), dtype=np.complex128 )
    for ti,t in enumerate( tGRID ):
        print("Time %1.3f a.u. of %1.3f" % (t, tGRID[-1]))
        psi_t[ti,:] = np.exp( -1j * E[:] * t ) * psi_0[:]
    return psi_t

def main():

    ### Get the energy basis by 
    ###   diagonalizing the Hamiltonian
    get_Globals()
    E, U = get_Energy_Basis()
    print( "QHO Energies:", E[:5] )

    ### Define the initial wavefunction 
    ###   in the basis of H (energy/eigen basis)
    
    # OPTION 1: Choose psi0 = QHO Ground State
    psi_0 = np.zeros( len(E) )
    psi_0[0]  = 1.0 
    psi_t = propagate( E, psi_0 ) # name is initial condition for saving files
    plt.plot( tGRID, psi_t[:,0].real, label="REAL" )
    plt.plot( tGRID, psi_t[:,0].imag, label="IMAG" )
    plt.plot( tGRID, np.abs(psi_t[:,0]), label="ABS" )
    plt.legend()
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Wavefunction, $C_n (t) = \\langle E_n | \\psi(t) \\rangle$", fontsize=15)
    plt.savefig(f"{DATA_DIR}/psi_t__E0.jpg", dpi=300)
    plt.clf()

    # OPTION 2: Choose psi0 as super-position of 
    #           ground and first excited state
    psi_0 = np.zeros( len(E), dtype=np.complex128 )
    psi_0[0] = 1 /np.sqrt(2) 
    psi_0[1] = 1 /np.sqrt(2) 
    psi_t = propagate( E, psi_0 )
    plt.plot( tGRID, psi_t[:,0].real,    "-",  c='black', lw=2, label="Re [$C_0$]" )
    plt.plot( tGRID, psi_t[:,0].imag,    "-",  c='black', lw=4, label="Im [$C_0$]" )
    plt.plot( tGRID, np.abs(psi_t[:,0]), "-",  c='black', lw=6, label="Abs[$C_0$]" )
    plt.plot( tGRID, psi_t[:,1].real,    "--", c='red',   lw=2, label="Re [$C_1$]" )
    plt.plot( tGRID, psi_t[:,1].imag,    "--", c='red',   lw=4, label="Im [$C_1$]" )
    plt.plot( tGRID, np.abs(psi_t[:,1]), "--", c='red',   lw=6, label="Abs[$C_1$]" )
    plt.legend()
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Wavefunction, $C_n (t) = \\langle E_n | \\psi(t) \\rangle$", fontsize=15)
    plt.savefig(f"{DATA_DIR}/psi_t__E0_E1.jpg", dpi=300)
    plt.clf()


def get_Energy_Basis():
    def get_V_x():
        return np.diag( 0.5 * xGRID**2 )

    def get_T_DVR():
        T = np.zeros((Nx,Nx))
        for i in range( Nx ):
            T[i,i] = np.pi**2 / 3
            for j in range( i+1, Nx ):
                T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
                T[j,i] = T[i,j]
        return T  / 2 / dx**2
    H = get_T_DVR() + get_V_x()
    E,U = np.linalg.eigh(H)
    return E,U

if ( __name__ == "__main__" ):
    main()

