import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

DATA_DIR = "2_Momentum_Space_Representation"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)


def get_Globals():
    global Nx, dx, xGRID, XMIN, XMAX
    Nx    =  101
    XMIN  = -10
    XMAX  =  10
    xGRID = np.linspace( XMIN, XMAX, Nx )
    dx    = xGRID[1] - xGRID[0]

    global kGRID, dk
    kGRID = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    kGRID = np.roll( kGRID, Nx//2 )
    dk    = kGRID[1] - kGRID[0]


def get_FFT( f_x ):
    f_k = np.fft.fft( f_x, norm='ortho' ) / np.sqrt( Nx )
    return f_k

def get_iFFT( f_x ):
    f_k = np.fft.ifft( f_x, norm='ortho' ) / np.sqrt( Nx )
    return f_k

def get_V_k():
    def get_V_x():
        return 0.5 * xGRID**2

    V_x = get_V_x()
    V_k = get_FFT( V_x )

    V_k_MAT = np.zeros( (Nx, Nx), dtype=complex )
    for j in range( Nx ):
        for k in range( Nx ):
            V_k_MAT[j,k] = V_k[j-k]
            
    plt.imshow( np.abs(V_k_MAT), origin='lower' )
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/V_k.jpg", dpi=300)
    plt.clf()

    return V_k_MAT

def get_T_k():
    T = np.diag( kGRID**2 ) / 2
    return T

def plot_wavefunctions( U ):

    # Momentum Space
    for state in range( 2 ):
        plt.plot( kGRID, U[:,state].real, "-",  label=f"State {state}" )
        plt.plot( kGRID, U[:,state].imag, "--" )
    plt.legend()
    plt.savefig(f"{DATA_DIR}/PSI_k.jpg", dpi=300)
    plt.clf()

    for state in range( 2 ):
        plt.plot( xGRID, np.abs(get_iFFT(U[:,state])), label=f"State {state}" )
    plt.legend()
    plt.savefig(f"{DATA_DIR}/PSI_x.jpg", dpi=300)
    plt.clf()


def main():
    get_Globals()

    V_k = get_V_k()
    T_k = get_T_k()
    H   = V_k + T_k

    E, U = np.linalg.eigh( H )
    print( "", E[:10] )
    plot_wavefunctions( U )

if ( __name__ == "__main__" ):
    main()