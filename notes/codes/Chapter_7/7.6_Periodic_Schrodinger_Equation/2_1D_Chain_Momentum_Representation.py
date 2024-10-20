import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import subprocess as sp
import scipy
from numba import njit

DATA_DIR = "2_1D_Chain_Momentum_Representation"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)


def get_Globals():
    global Nx, dx, xGRID, Lx
    Nx    = 256 # Choose to be a power of 2 for FFT speed boost
    Lx    = 30.0 # Lattice Constant
    xGRID = np.linspace( -Lx/2, Lx/2, Nx )
    if ( (xGRID == 0).any() ): xGRID += 1e-4 # erf(x)/x is singular at x=0
    dx    = xGRID[1] - xGRID[0]

    global kGRID
    kGRID = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    kGRID = np.roll( kGRID, Nx//2 )

    global KPOINT_LIST, NKPTS, NBZS, NBANDS
    NKPTS       = 201 # Number of K-Points
    NBZS        = 2 # Number of Brillouin Zones to compute
    NBANDS      = 3 # Number of Bands to plot
    KPOINT_LIST = np.linspace( -NBZS*np.pi/Lx, NBZS*np.pi/Lx, NKPTS )

    global cmap
    cmap = matplotlib.colormaps['copper']

def get_V_x():
    r0 = 10 # 0.1,1, 100
    Vx  = -scipy.special.erf( xGRID / r0 ) / xGRID #/ Lx
    return Vx

def get_FFT( f_x ):
    f_k = np.fft.fft( f_x, norm='ortho' ) / np.sqrt( Nx )
    return f_k

def get_iFFT( f_x ):
    f_k = np.fft.ifft( f_x, norm='ortho' ) / np.sqrt( Nx )
    return f_k

def get_V_k( doPLOT = False):

    V_x = get_V_x()
    V_k = get_FFT( V_x )

    V_k_MAT = np.zeros( (Nx, Nx), dtype=complex )
    for j in range( Nx ):
        for k in range( Nx ):
            V_k_MAT[j,k] = V_k[j-k]

    if ( doPLOT == True ):
        plt.imshow( np.abs(V_k_MAT), cmap='Greys', origin='lower' )
        plt.colorbar(pad=0.01)
        plt.savefig(f"{DATA_DIR}/V_k.jpg", dpi=300)
        plt.clf()
    return V_k_MAT

@njit
def get_T_k( K_POINT=0.0 ):
    T = (kGRID + K_POINT)**2 / 2
    return np.diag(T)

@njit
def eigh(H):
    return np.linalg.eigh(H)

def do_Gamma_Point():
    H = get_T_k( K_POINT=0.0 ) + get_V_k( doPLOT=True )
    E, U = eigh(H)
    Ux   = np.zeros_like( U )
    for state in range( 10 ):
        Ux[:,state]  = np.abs(get_iFFT( U[:,state] ))
    plt.plot( xGRID, get_V_x(), c='black', lw=8, alpha=0.5, label="V(x)" )
    plt.plot( xGRID, Ux[:,0] + E[0], label="$|\\phi_{K=0,n=0}\\rangle$" )
    plt.plot( xGRID, Ux[:,1] + E[1], label="$|\\phi_{K=0,n=1}\\rangle$" )
    plt.plot( xGRID, Ux[:,2] + E[2], "--", label="$|\\phi_{K=0,n=2}\\rangle$" )
    plt.xlabel("Position, $x$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E(K)$ (a.u.)", fontsize=15)
    plt.legend()
    plt.tight_layout()
    #plt.ylim(0,2)
    plt.savefig(f"{DATA_DIR}/V_x.jpg", dpi=300)
    plt.clf()

def do_Band_Structure():

    E_LIST      = np.zeros( (NKPTS, Nx) )
    U_LIST      = np.zeros( (NKPTS, Nx, Nx), dtype=np.complex128 )
    for iK, K in enumerate( KPOINT_LIST ):
        print("Working on KPT %d" % iK)
        H          = get_T_k( K_POINT=K ) + get_V_k()
        E, U       = eigh(H)
        E_LIST[iK] = E
        U_LIST[iK] = U

    fig, ax = plt.subplots()
    for state in range( NBANDS ):
        plt.scatter( KPOINT_LIST/np.pi*Lx, E_LIST[:,state] - np.min(E_LIST[:,0]), c=cmap(np.abs(KPOINT_LIST)/KPOINT_LIST[-1]) )
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(label="K-Point, $\\frac{K L}{\\pi}$ (a.u.)", size=15)
    plt.xlabel("K-Point, $\\frac{K L}{\\pi}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E(K)$ (a.u.)", fontsize=15)
    plt.plot( KPOINT_LIST/np.pi*Lx, 0.5 * KPOINT_LIST**2, "--", c="grey", label="Independent Electron Model" )
    for BZ in range( 1, NBANDS ):
        plt.plot( KPOINT_LIST/np.pi*Lx, 0.5 * (KPOINT_LIST + 2*BZ*np.pi/Lx)**2, "--", c="grey" )
        plt.plot( KPOINT_LIST/np.pi*Lx, 0.5 * (KPOINT_LIST - 2*BZ*np.pi/Lx)**2, "--", c="grey" )
    plt.legend()
    plt.xlim( KPOINT_LIST[0]/np.pi*Lx, KPOINT_LIST[-1]/np.pi*Lx )
    plt.ylim(0,np.max(E_LIST[:,NBANDS-1] - np.min(E_LIST[:,0])))
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/E_k.jpg", dpi=300)
    plt.clf()

    NSKIP = NKPTS // 10
    for state in range( 4 ):
        fig, ax = plt.subplots()
        for iK, K in enumerate( KPOINT_LIST[::NSKIP] ):
            Ux = get_iFFT( U_LIST[iK,:,state] )
            scale = np.max(np.abs(Ux)) - np.min(np.abs(Ux))
            if ( K >= 0 ):
                plt.plot( xGRID, np.abs(Ux), c=cmap(np.abs(K)/KPOINT_LIST[-1]) )
        sm = plt.cm.ScalarMappable(cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(label="K-Point, $\\frac{K L}{\\pi}$ (a.u.)", size=15)
        plt.xlabel("Position, $x$ (a.u.)", fontsize=15)
        plt.ylabel("Energy, $E(K)$ (a.u.)", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{DATA_DIR}/PSI_x_{state}_KPTS.jpg", dpi=300)
        plt.clf()


def main():
    get_Globals()

    do_Gamma_Point()

    do_Band_Structure()








if ( __name__ == "__main__" ):
    main()