import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import subprocess as sp
from numba import njit

import imageio.v2 as imageio
from pygifsicle import optimize as gifOPT # This needs to be installed somewhere
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "8.4_TDSE_photo_activation_classical"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global make_movies
    make_movies = False  # This takes time. If True, will make movies of wavepacket. 
                        # If false, will generate pictures of product population.

    global Nx, dx, xGRID
    XMIN  = -10.0
    XMAX  = 10.0
    Nx    = 1000
    xGRID = np.linspace(XMIN, XMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    global dt, tGRID, NSTEPS
    tMAX  = 2000.0 # 20.0
    dt    = 1
    tGRID = np.arange(0, tMAX+dt, dt)
    NSTEPS = len(tGRID)

@njit()
def propagate( E_HAM, U_HAM, psi_0, EFIELD ):
    psi_t      = np.zeros( (NSTEPS,Nx), dtype=np.complex128 )
    psi_t[0,:] = psi_0
    EDIFF      = E_HAM - E_HAM[0] + 0.0j
    U_HAM      = U_HAM.astype(np.complex128)
    for ti in range( 1, NSTEPS ):
        if ( ti%100 == 0 ): print(ti, "of", NSTEPS)
        t           = tGRID[ti]
        psi_t[ti,:] = U_HAM @ psi_t[ti-1,:] # < E_n | psi (t) > --> < x | psi (t) >
        psi_t[ti,:] = np.exp( -1j * EFIELD[ti,:] * dt )  * psi_t[ti,:]
        psi_t[ti,:] = U_HAM.T @ psi_t[ti,:] # < x | psi (t) > --> < E_n | psi (t) >
        psi_t[ti,:] = np.exp( -1j * EDIFF * dt ) * psi_t[ti,:]
        # psi_t[ti,:] = np.einsum("xE,E->x", U_HAM, psi_t[ti-1,:]) # < E_n | psi (t) > --> < x | psi (t) >
        # psi_t[ti,:] = np.exp( -1j * EFIELD[ti,:] * dt )  * psi_t[ti,:]
        # psi_t[ti,:] = np.einsum("xE,x->E", U_HAM, psi_t[ti,:]) # < x | psi (t) > --> < E_n | psi (t) >
        # psi_t[ti,:] = np.exp( -1j * EDIFF * dt ) * psi_t[ti,:]
    return psi_t

def left_state_with_LASER( E_HAM, U_HAM, EFIELD ):
    ### Define the initial wavefunction 
    psi_0      = np.zeros( Nx, dtype=np.complex128 )
    psi_0      = U_HAM[:,0] + U_HAM[:,1] # Start in the left well
    if ( np.sum(np.abs(psi_0[:Nx//2])**2) < 0.1 ): psi_0 = U_HAM[:,0] - U_HAM[:,1]
    psi_0      = psi_0 / np.linalg.norm(psi_0) # Normalize the wavefnuction
    psi_0      = np.einsum("xE,x->E", U_HAM, psi_0) # < x | psi (t) >  -->  < E_n | psi (t) >
    psi_t      = propagate( E_HAM, U_HAM, psi_0, EFIELD ) # < E_n | psi (0) >  -->  < E_n | psi (t) >
    E_t        = np.einsum("tE,E,tE->t", psi_t.conj(), E_HAM, psi_t).real
    psi_t      = np.einsum("xE,tE->tx", U_HAM[:,:], psi_t[:,:]) # < E_n | psi (t) >  -->  < x | psi (t) >
    #X1_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID, psi_t).real # <x>
    #X2_t       = np.einsum("tx,x,tx->t", psi_t.conj(), xGRID**2, psi_t).real # <x^2>
    return psi_t, E_t#, X1_t, X2_t

def main():
    get_Globals()

    ### Get the energy basis by 
    ###   diagonalizing the Hamiltonian
    E_HAM, U_HAM = get_Energy_Basis()
    print( "Hamiltonian Eigen-Energies:\n", E_HAM[:5] )

    # Define the electric field
    # E0     = 0.1
    # FREQ   = E_HAM[2] - E_HAM[0] # Choose resonance condition
    # EFIELD = E0 * np.cos( FREQ * tGRID[:,None] ) * xGRID[None,:] # (t,x)
    # psi_t, E_t = left_state_with_LASER( E_HAM, U_HAM, EFIELD )
    # make_x_movie(psi_t, E_t, EFIELD, name="left_state_with_LASER")
    # make_E_movie(np.einsum("xE,tx->tE", U_HAM, psi_t), name="left_state_with_LASER")

    # ### Left State with Varying LASER frequency
    RESONANCE_STATE = 2 # (0,1), (2,3), (4,5), (6,7), (8,9), ...
    E0_list   = np.array([1e-5,1e-4,1e-3,1e-2,1e-1])
    FREQ_list = np.linspace(E_HAM[RESONANCE_STATE]-E_HAM[0]-0.1, E_HAM[RESONANCE_STATE]-E_HAM[0]+0.1, 11)
    # Normalize the colors
    color_list = plt.get_cmap("brg")(np.linspace(0,1,len(FREQ_list)))
    RATES_LIN  = np.zeros( (len(FREQ_list), len(E0_list)) )
    RATES_EXP  = np.zeros( (len(FREQ_list), len(E0_list)) )
    x_mask     = xGRID > 0.0
    for E0i,E0 in enumerate(E0_list):
        for FREQi,FREQ in enumerate(FREQ_list):
            print("Working on FREQ", FREQi, "of", len(FREQ_list))
            # Define the electric field
            EFIELD = E0 * np.cos( FREQ * tGRID[:,None] ) * xGRID[None,:] # E(t,x)
            psi_t, E_t           = left_state_with_LASER( E_HAM, U_HAM, EFIELD )
            density              = np.real( psi_t[:,x_mask].conj() * psi_t[:,x_mask] )
            product_population   = np.sum(density, axis=1)
            RATES_LIN[FREQi,E0i] = np.average( product_population[1:] / tGRID[1:] )
            RATES_EXP[FREQi,E0i] = np.average( -np.log(1 - product_population[1:]) / tGRID[1:] )
            print("RATES (LIN,EXP): (%1.2e, %1.2e) " % (RATES_LIN[FREQi,E0i], RATES_EXP[FREQi,E0i]) )
            if ( make_movies == False ):
                plt.semilogy( tGRID, product_population, "-", c=color_list[FREQi], lw=2, label="$\\frac{\\omega_\\mathrm{c}}{E_%d - E_0}$ = %1.2f" % (RESONANCE_STATE,FREQ/(E_HAM[RESONANCE_STATE]-E_HAM[0])) * (len(FREQ_list) <= 11) )
            elif ( make_movies == True ):
                make_x_movie(psi_t, E_t, EFIELD, name="left_state_with_LASER_FREQ_WC_%1.4f_E0_%1.2e" % (FREQ,E0))
        if ( make_movies == False ):
            plt.xlabel("Time (a.u.)", fontsize=15)
            plt.ylabel("Product Population", fontsize=15)
            plt.legend()
            plt.ylim(1e-8,1)
            #plt.ylim(0,1)
            plt.tight_layout()
            plt.savefig("%s/product_population_varying_LASER_FREQ_E0_%1.2e.png" % (DATA_DIR,E0),dpi=300)
            plt.clf()
            plt.close()

    color_list = plt.get_cmap("brg")(np.linspace(0,1,len(E0_list)))
    for E0i,E0 in enumerate(E0_list):
        plt.semilogy( FREQ_list / (E_HAM[RESONANCE_STATE] - E_HAM[0]), RATES_EXP[:,E0i], "-o",  c=color_list[E0i], label="$E_0$ = %1.2e" % (E0) )
        plt.semilogy( FREQ_list / (E_HAM[RESONANCE_STATE] - E_HAM[0]), RATES_LIN[:,E0i], "--o", c=color_list[E0i] )
    plt.xlabel("LASER Frequency ($\\frac{\\omega_\\mathrm{c}}{E_%d - E_0}$)" % (RESONANCE_STATE), fontsize=15)
    plt.ylabel("Product Rate", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/product_rate_varying_LASER_FREQ_STRENGTH.png",dpi=300)
    plt.clf()
    plt.close()



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




def make_x_movie( psi_t, E_t, EFIELD, name="" ):

    def make_frame( psi, E, t, FIELD ):
        plt.plot( xGRID, Vx + FIELD, "-", c='black', lw=8, alpha=0.5, label="V(x)" )
        norm = np.max(np.abs(psi))
        plt.plot( xGRID, 2 * np.abs(psi)/norm  + E, "-",  c='black', lw=2, label="ABS" )
        plt.plot( xGRID, 2 * np.real(psi)/norm + E, "-",  c='red', lw=2, label="RE" )
        plt.plot( xGRID, 2 * np.imag(psi)/norm + E, "-",  c='green', lw=2, label="IM" )
        plt.legend(loc=1)
        plt.xlim(-4,4)
        plt.ylim(-5, 30 )
        plt.title("Energy = %1.2f, time = %1.2f" % (E, t), fontsize=15)
        plt.xlabel("Position (a.u.)", fontsize=15)
        plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"DUMMY.jpg",dpi=70)
        plt.clf()
        plt.close()


    movieNAME = f"{DATA_DIR}/movie_{name}_X.gif"
    #NSKIP     = len(tGRID) #// NFRAMES
    with imageio.get_writer(movieNAME, loop=4, mode='I', fps=20) as writer: # Get a writer object
        #for frame in range( 0, NSTEPS, NSKIP ):
        for frame in range( 0, NSTEPS, 5 ):
            #print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1)//NSKIP, NSTEPS//NSKIP) )
            print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1), NSTEPS) )
            make_frame( psi_t[frame], E_t[frame], tGRID[frame], EFIELD[frame] )
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

