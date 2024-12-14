import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import subprocess as sp

import imageio.v2 as imageio
from pygifsicle import optimize as gifOPT # This needs to be installed somewhere
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "8.3_TDSE_Symmetric_DoubleWell"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global Nx, dx, xGRID
    XMIN  = -10.0
    XMAX  = 10.0
    Nx    = 2000
    xGRID = np.linspace(XMIN, XMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    global dt, tGRID, NSTEPS
    tMAX  = 1.0 # 20.0
    dt    = 0.01
    tGRID = np.arange(0, tMAX+dt, dt)
    NSTEPS = len(tGRID)

def propagate( E_HAM, psi_0, name="" ):
    ### Propagate the wavefunction in time
    psi_t = np.zeros( (len(tGRID), len(E_HAM)), dtype=np.complex128 )
    psi_t[0,:] = psi_0[:]
    for ti,t in enumerate( tGRID ):
        if (ti == 0): continue
        # Rather than diag(E), we can just do element-wise vector multiplication
        # This is faster since all off-diagonal elements of the diag(E) are zero
        psi_t[ti,:] = np.exp( -1j * E_HAM[:] * t ) * psi_0[:]
    
    return psi_t

def ground_state( E_HAM, U_HAM ):
    ### Define the initial wavefunction 
    psi_0      = np.zeros( Nx, dtype=np.complex128 )
    psi_0      = U_HAM[:,0] # np.exp( -(xGRID+2)**2 / 2 / 1**2 )
    psi_0      = psi_0 / np.linalg.norm(psi_0)
    psi_0      = np.einsum("xE,x->E", U_HAM, psi_0) # < x | psi (t) >  -->  < E_n | psi (t) >
    psi_t      = propagate( E_HAM, psi_0 ) # < E_n | psi (0) >  -->  < E_n | psi (t) >
    E_t        = np.einsum("tE,E,tE->t", psi_t.conj(), E_HAM, psi_t).real
    psi_t      = np.einsum("xE,tE->tx", U_HAM[:,:], psi_t[:,:]) # < E_n | psi (t) >  -->  < x | psi (t) >
    X1_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID, psi_t).real
    X2_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID**2, psi_t).real
    return psi_t, E_t, X1_t, X2_t

def left_state( E_HAM, U_HAM ):
    ### Define the initial wavefunction 
    psi_0      = np.zeros( Nx, dtype=np.complex128 )
    psi_0      = U_HAM[:,0] - U_HAM[:,1]  # np.exp( -(xGRID+2)**2 / 2 / 1**2 ) # SIG = 0.5 is the exact ground state
    psi_0      = psi_0 / np.linalg.norm(psi_0)
    psi_0      = np.einsum("xE,x->E", U_HAM, psi_0) # < x | psi (t) >  -->  < E_n | psi (t) >
    psi_t      = propagate( E_HAM, psi_0 ) # < E_n | psi (0) >  -->  < E_n | psi (t) >
    E_t        = np.einsum("tE,E,tE->t", psi_t.conj(), E_HAM, psi_t).real
    psi_t      = np.einsum("xE,tE->tx", U_HAM[:,:], psi_t[:,:]) # < E_n | psi (t) >  -->  < x | psi (t) >
    X1_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID, psi_t).real
    X2_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID**2, psi_t).real
    return psi_t, E_t, X1_t, X2_t

def left_state_with_momentum( E_HAM, U_HAM, p0=0.0 ):
    ### Define the initial wavefunction 
    psi_0      = np.zeros( Nx, dtype=np.complex128 )
    #psi_0      = U_HAM[:,0] - U_HAM[:,1]  # np.exp( -(xGRID+2)**2 / 2 / 1**2 ) # SIG = 0.5 is the exact ground state
    psi_0      = np.exp( -(xGRID + 2)**2 / 2 / 0.5**2 )  # np.exp( -(xGRID+2)**2 / 2 / 1**2 ) # SIG = 0.5 is the exact ground state
    psi_0      = psi_0[:] * np.exp( 1j * xGRID[:] * p0 ) # Add momentum, 1j and p0>0 gives positive momentum
    psi_0      = psi_0 / np.linalg.norm(psi_0)
    psi_0      = np.einsum("xE,x->E", U_HAM, psi_0) # < x | psi (t) >  -->  < E_n | psi (t) >
    psi_t      = propagate( E_HAM, psi_0 ) # < E_n | psi (0) >  -->  < E_n | psi (t) >
    E_t        = np.einsum("tE,E,tE->t", psi_t.conj(), E_HAM, psi_t).real
    psi_t      = np.einsum("xE,tE->tx", U_HAM[:,:], psi_t[:,:]) # < E_n | psi (t) >  -->  < x | psi (t) >
    X1_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID, psi_t).real
    X2_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID**2, psi_t).real
    return psi_t, E_t, X1_t, X2_t

def main():
    get_Globals()

    ### Get the energy basis by 
    ###   diagonalizing the Hamiltonian
    E_HAM, U_HAM = get_Energy_Basis()
    print( "Hamiltonian Eigen-Energies:\n", E_HAM[:5] )

    # ### Ground State
    # psi_t, E_t, X1_t, X2_t = ground_state( E_HAM, U_HAM )
    # make_x_movie(psi_t, E_t, name="ground_state")
    # make_E_movie(np.einsum("xE,tx->tE", U_HAM, psi_t), name="ground_state")
    # make_plot_X1_X2(X1_t, X2_t, name="ground_state")

    # ### Left State
    # psi_t, E_t, X1_t, X2_t = left_state( E_HAM, U_HAM )
    # make_x_movie(psi_t, E_t, name="left_state")
    # make_E_movie(np.einsum("xE,tx->tE", U_HAM, psi_t), name="left_state")
    # make_plot_X1_X2(X1_t, X2_t, name="left_state")

    # ### Left State with Momentum
    # psi_t, E_t, X1_t, X2_t = left_state_with_momentum( E_HAM, U_HAM, p0=2.0 )
    # make_x_movie(psi_t, E_t, name="left_state_with_momentum")
    # make_E_movie(np.einsum("xE,tx->tE", U_HAM, psi_t), name="left_state_with_momentum")
    # make_plot_X1_X2(X1_t, X2_t, name="left_state_with_momentum")

    ### Left State with Varying Momentum
    momentum_list = np.arange(0, 10, 1)
    measure_function = np.zeros( Nx )
    measure_function[ xGRID > 0 ] = 1 # Measure the population in the right well
    for p0i,p0 in enumerate(momentum_list):
        print("Working on Momentum: ", p0)
        psi_t, E_t, X1_t, X2_t    = left_state_with_momentum( E_HAM, U_HAM, p0=p0 )
        density                   = np.abs(psi_t)**2
        product_population = np.einsum("tx,x->t", density[:,:], measure_function[:] )
        if ( E_t[0] < Vx[Nx//2] ): # Plot solid if in the E < Barrier, else dashed
            plt.semilogy( tGRID, product_population, "-", lw=2, label="$p_0$ = %1.2f (E < Barrier)" % (p0) )
        else:
            plt.semilogy( tGRID, product_population, "--", lw=2, label="$p_0$ = %1.2f (E > Barrier)" % (p0) )
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Product Population", fontsize=15)
    plt.legend()
    plt.ylim(1e-6,1)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/product_population_varying_p0.png",dpi=300)
    plt.clf()
    plt.close()


    ##### MAKE THE MOVIES #####
    for p0i,p0 in enumerate(momentum_list):
        print("Working on Momentum: ", p0)
        psi_t, E_t, X1_t, X2_t    = left_state_with_momentum( E_HAM, U_HAM, p0=p0 )
        if ( E_t[0] < 30 ):
           make_x_movie(psi_t, E_t, name="left_state_with_momentum_p0_%1.2f" % (p0))


def get_Energy_Basis():
    def get_V_x():
        global Vx
        Vx  = ( xGRID**2 - 4 ) ** 2
        Vx -= np.min(Vx)
        return np.diag( Vx )

    def get_T_DVR():
        T = np.zeros((Nx,Nx))
        for i in range( Nx ):
            T[i,i] = np.pi**2 / 3
            for j in range( i+1, Nx ):
                T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
                T[j,i] = T[i,j]
        return T  / 2 / dx**2
    T = get_T_DVR()
    V = get_V_x()
    H = T + V
    E,U = np.linalg.eigh(H)
    U = U

    ### Do phase correction
    SUM = np.sum( U[:,0] )
    if ( SUM < 0 ): U *= -1

    # Plot wavefunctions of lowest 5 states shifted by their energy
    plt.plot( xGRID, np.diag( V ), "-", c='black', lw=8, alpha=0.5, label="V(x)" )
    for state in range(10):
        norm = U[:,state]/np.max(np.abs(U[:,state]))
        if ( state%2 == 0 ):
            plt.plot( xGRID, E[state] + norm, "-", lw=2, label="$\\phi_%d$" % (state) )
        else:
            plt.plot( xGRID, E[state] + norm, "--", lw=2, label="$\\phi_%d$" % (state) )
    plt.legend()
    plt.xlim(-5,5)
    plt.ylim(0,1.1*E[10])
    plt.xlabel("Position (a.u.)", fontsize=15)
    plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/Vx_WFNs.jpg",dpi=300)
    plt.clf()
    plt.close()
    return E,U




def make_x_movie( psi_t, E_t, name="" ):

    def make_frame( psi, E, t ):
        plt.plot( xGRID, Vx, "-", c='black', lw=8, alpha=0.5, label="V(x)" )
        norm = np.max(np.abs(psi))
        plt.plot( xGRID, 2 * np.abs(psi)/norm  + E, "-",  c='black', lw=2, label="ABS" )
        plt.plot( xGRID, 2 * np.real(psi)/norm + E, "-",  c='red', lw=2, label="RE" )
        plt.plot( xGRID, 2 * np.imag(psi)/norm + E, "-",  c='green', lw=2, label="IM" )
        plt.legend(loc=1)
        plt.xlim(-4,4)
        plt.ylim(0, 30 )
        plt.title("Energy = %1.2f, time = %1.2f" % (E, t), fontsize=15)
        plt.xlabel("Position (a.u.)", fontsize=15)
        plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"DUMMY.jpg",dpi=70)
        plt.clf()


    movieNAME = f"{DATA_DIR}/movie_{name}_X.gif"
    #NSKIP     = len(tGRID) #// NFRAMES
    with imageio.get_writer(movieNAME, loop=4, mode='I', fps=20) as writer: # Get a writer object
        #for frame in range( 0, NSTEPS, NSKIP ):
        for frame in range( 0, NSTEPS ):
            #print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1)//NSKIP, NSTEPS//NSKIP) )
            print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1), NSTEPS) )
            make_frame( psi_t[frame], E_t[frame], tGRID[frame] )
            image = imageio.imread( "DUMMY.jpg" ) # Read JPEG file
            writer.append_data(image) # Write JPEG file (to memory at first; then printed at end)
    sp.call("rm DUMMY.jpg", shell=True)
    gifOPT(movieNAME) # This will compress the GIF movie by at least a factor of two/three. With this: ~750 frames --> 80 MB

def make_E_movie( psi_t, name="" ):

    def make_frame( psi ):
        norm = np.max(np.abs(psi))
        plt.plot( np.arange(Nx), np.abs(psi)/norm , "-",  c='black', lw=2 )
        plt.plot( np.arange(Nx), np.real(psi)/norm , "-",  c='red', lw=2 )
        plt.plot( np.arange(Nx), np.imag(psi)/norm , "-",  c='green', lw=2 )
        plt.xlabel("Position (a.u.)", fontsize=15)
        plt.ylabel("Wavefunction, $|\\psi_n(t) = \\langle n | \\psi(t) \\rangle|^2$", fontsize=15)
        plt.xlim(0,10)
        plt.ylim(-1,1)
        plt.tight_layout()
        plt.savefig(f"DUMMY.jpg",dpi=70)
        plt.clf()

    movieNAME = f"{DATA_DIR}/movie_{name}_E.gif"
    #NSKIP     = len(tGRID) #// NFRAMES
    with imageio.get_writer(movieNAME, loop=4, mode='I', fps=20) as writer: # Get a writer object
        #for frame in range( 0, NSTEPS, NSKIP ):
        for frame in range( 0, NSTEPS ):
            #print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1)//NSKIP, NSTEPS//NSKIP) )
            print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1), NSTEPS) )
            make_frame( psi_t[frame] )
            image = imageio.imread( "DUMMY.jpg" ) # Read JPEG file
            writer.append_data(image) # Write JPEG file (to memory at first; then printed at end)
    sp.call("rm DUMMY.jpg", shell=True)
    gifOPT(movieNAME) # This will compress the GIF movie by at least a factor of two/three. With this: ~750 frames --> 80 MB


def make_plot_X1_X2( X1_t, X2_t, name="" ):
    plt.plot( tGRID, X1_t, "-", c='black', lw=2, label="$\\langle \\hat{X} \\rangle$" )
    plt.plot( tGRID, np.sqrt(X2_t - X1_t**2), "-", c='red', lw=2, label="$\\sigma_X = \\sqrt{\\langle \\hat{X}^2 \\rangle - \\langle \\hat{X} \\rangle^2}$" )
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Expectation Values", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/X1_X2_{name}.png",dpi=300)
    plt.clf()



if ( __name__ == "__main__" ):
    main()

