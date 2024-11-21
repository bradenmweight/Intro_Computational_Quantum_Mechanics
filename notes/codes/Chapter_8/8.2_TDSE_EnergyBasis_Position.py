import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import subprocess as sp

import imageio.v2 as imageio
from pygifsicle import optimize as gifOPT # This needs to be installed somewhere
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "8.2_TDSE_EnergyBasis_Position"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global Nx, dx, xGRID
    XMIN  = -10.0
    XMAX  = 10.0
    Nx    = 2000
    xGRID = np.linspace(XMIN, XMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    global dt, tGRID, NSTEPS
    tMAX  = 8.0
    dt    = 0.05
    tGRID = np.arange(0, tMAX+dt, dt)
    NSTEPS = len(tGRID)

def propagate( E_QHO, psi_0, name="" ):
    ### Propagate the wavefunction in time
    psi_t = np.zeros( (len(tGRID), len(E_QHO)), dtype=np.complex128 )
    E_t   = np.zeros( len(tGRID) )
    for ti,t in enumerate( tGRID ):
        # Rather than diag(E), we can just do element-wise vector multiplication
        # This is faster since all off-diagonal elements of the diag(E) are zero
        psi_t[ti,:] = np.exp( -1j * E_QHO[:] * t ) * psi_0[:]
        E_t[ti]     = np.average( E_QHO[:] * psi_t[ti,:] ).real
    return psi_t, E_t

def moving_Gaussian( E_QHO, U_QHO ):
    ### Define the initial wavefunction 
    psi_0      = np.zeros( Nx, dtype=np.complex128 )
    psi_0      = np.exp( -(xGRID-1)**2 / 2 / 0.5**2 )
    psi_0      = psi_0 / np.linalg.norm(psi_0)
    psi_0      = np.einsum("xE,x->E", U_QHO, psi_0)
    psi_t, E_t = propagate( E_QHO, psi_0 ) # < E_n | psi (t) >
    psi_t      = np.einsum("xE,tE->tx", U_QHO[:,:], psi_t[:,:]) # < x | psi (t) >
    return psi_t, E_t

def stationary_Gaussian( E_QHO, U_QHO ):
    ### Define the initial wavefunction 
    psi_0      = np.exp( -xGRID**2 / 2 / 0.4**2 ) # SIG = 0.5 is the exact ground state
    psi_0      = psi_0 / np.linalg.norm(psi_0)
    psi_0      = np.einsum("xE,x->E", U_QHO, psi_0)
    psi_t, E_t = propagate( E_QHO, psi_0 ) # < E_n | psi (t) >
    psi_t      = np.einsum("xE,tE->tx", U_QHO[:,:], psi_t[:,:]) # < x | psi (t) >
    return psi_t, E_t

def main():
    get_Globals()

    ### Get the energy basis by 
    ###   diagonalizing the Hamiltonian
    E_QHO, U_QHO = get_Energy_Basis()
    print( "QHO Energies:", E_QHO[:5] )

    ### Example #1 -- A translating Gaussian
    psi_t, E_t = moving_Gaussian( E_QHO, U_QHO )
    make_movie(psi_t, E_t, name="moving_Gaussian")

    ### Example #2 -- A stationary Gaussian
    psi_t, E_t = stationary_Gaussian( E_QHO, U_QHO )
    make_movie(psi_t, E_t, name="stationary_Gaussian")








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




def make_movie( psi_t, E_t, name="" ):

    def make_frame( psi, E ):
        plt.plot( xGRID, 0.500 * xGRID**2, "-", c='black', lw=8, alpha=0.5, label="V(x)" )
        plt.plot( xGRID, np.abs(psi)/np.max(np.abs(psi)) + E, "-",  c='black', lw=2, label="ABS" )
        plt.legend()
        plt.xlim(-5,5)
        plt.ylim(0,1.5)
        plt.xlabel("Position (a.u.)", fontsize=15)
        plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"DUMMY.jpg",dpi=70)
        plt.clf()


    movieNAME = f"{DATA_DIR}/movie_{name}.gif"
    #NSKIP     = len(tGRID) #// NFRAMES
    with imageio.get_writer(movieNAME, loop=4, mode='I', fps=20) as writer: # Get a writer object
        #for frame in range( 0, NSTEPS, NSKIP ):
        for frame in range( 0, NSTEPS ):
            #print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1)//NSKIP, NSTEPS//NSKIP) )
            print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1), NSTEPS) )
            make_frame( psi_t[frame], E_t[frame] )
            image = imageio.imread( "DUMMY.jpg" ) # Read JPEG file
            writer.append_data(image) # Write JPEG file (to memory at first; then printed at end)
    sp.call("rm DUMMY.jpg", shell=True)
    gifOPT(movieNAME) # This will compress the GIF movie by at least a factor of two/three. With this: ~750 frames --> 80 MB





if ( __name__ == "__main__" ):
    main()

