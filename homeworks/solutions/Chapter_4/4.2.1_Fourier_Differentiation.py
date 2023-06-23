import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "4.2.1_PLOTS"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_f_x():
    global p0_LIST
    Nx = 501
    x = np.linspace( -30,30,Nx )
    
    # Function 1
    f_x = np.exp( -x**2/2 )

    # Function 2
    #f_x = np.cos( 2*x ) * np.exp( -x**2/20 )
    
    # Function 3
    #f_x = np.exp( -np.abs(x) / 2 ) * np.cos( 2 * np.pi * x ) * np.cos( np.pi * (x-2) )

    return x, f_x

def f_1_x__ANALYTIC( x ):

    # Function 1
    return -x * np.exp(-x**2/2)

    # Function 2
    #return (2*np.sin(2*x) + 2*x/10 * np.cos(2*x) ) * np.exp( -x**2/10 )

    # Function 3
    #T1 = -1*(x*np.exp(-np.abs(x)/2) * np.cos(np.pi * (x-2)) * np.cos( 2 * np.pi * x)) / 2 / np.abs(x)
    #T2 = -np.pi * np.exp(-np.abs(x)/2) * np.sin(np.pi * (x-2)) * np.cos( 2 * np.pi * x)
    #T3 = -2*np.pi * np.exp(-np.abs(x)/2) * np.cos(np.pi * (x-2)) * np.sin( 2 * np.pi * x)
    #return T1 + T2 + T3

def plot_f_x(x,f_x):

    plt.plot( x, f_x, "-" )
    plt.xlim(-10,10)
    plt.xlabel("x",fontsize=15)
    plt.ylabel("Wavefunction",fontsize=15)
    plt.savefig(f"{DATA_DIR}/f_x.jpg",dpi=400)
    plt.clf()

def get_DFT( x, f_x ):

    dx = x[1] - x[0]
    Lx = x[1] - x[0]
    Nx = len(x)

    # Define the p-grid (reciprocal grid)
    dp   = 2 * np.pi / Lx
    pmax = np.pi / dx # This is not angular frequency
    p    = np.linspace( -pmax, pmax, Nx )

    f_p = np.zeros( Nx, dtype=complex )

    # Define the Fourier matrix, W, and operate
    n = np.arange(Nx, dtype=complex).reshape( (-1,1) )
    m = np.arange(Nx, dtype=complex).reshape( (1,-1) )
    W = np.exp( -2j*np.pi * (m-Nx//2) * (n-Nx//2) / Nx )

    f_p[:] = W @ f_x[:]
    f_p[:] *= dx / np.sqrt( 2 * np.pi )

    return p, f_p

def plot_f_p(p,f_p):

    plt.plot( p, f_p.real, "-", label="$f(p)$" )
    plt.plot( p, f_p.real * p, "-", label="$f(p) * p$" )
    plt.legend()
    #plt.xlim(-4,4)
    plt.xlabel("p",fontsize=15)
    plt.ylabel("Wavefunction",fontsize=15)
    plt.savefig(f"{DATA_DIR}/f_p.jpg",dpi=400)
    plt.clf()


def get_iDFT( p, f_p, nDERIV=0 ):

    dp = p[1] - p[0]
    Np = len(p)

    f_n_x = np.zeros( Np, dtype=complex )

    # Define the inverse Fourier matrix, W, and operate
    n = np.arange(Np, dtype=complex).reshape( (-1,1) )
    m = np.arange(Np, dtype=complex).reshape( (1,-1) )
    W = np.exp( -2j*np.pi * (m-Np//2) * (n-Np//2) / Np )

    f_n_x[:]  = W @ ( f_p[:] * (-1.0j*p)**nDERIV )
    f_n_x[:] *= dp / np.sqrt( 2 * np.pi )

    return f_n_x


def plot_f_x__f_p_x( x, f_x, f_n_x ):

    plt.plot( x, f_x, "-", label="$f(x)$" )
    plt.plot( x, f_n_x.real, "-", label="$\\frac{\partial^{n} f(x)}{\partial x^{n}}$ (NUM.)" )
    #plt.plot( x, f_n_x.imag, ":", label="$\\frac{\partial^{n} f(x)}{\partial x^{n}}$ (NUM.)" )
    plt.plot( x, f_1_x__ANALYTIC( x ), ":", label="$\\frac{\partial^{n} f(x)}{\partial x^{n}}$ (ANA.)" )
    
    plt.legend()
    plt.xlim(-10,10)
    plt.xlabel("x",fontsize=15)
    plt.ylabel("Wavefunction",fontsize=15)
    plt.savefig(f"{DATA_DIR}/f_n_x.jpg",dpi=400)
    plt.clf()

def main():
    x, f_x = get_f_x()
    plot_f_x(x,f_x)
    p, f_p = get_DFT(x,f_x)
    plot_f_p(p,f_p)
    f_n_x = get_iDFT(p, f_p,nDERIV=1) # n = order of derivative
    plot_f_x__f_p_x( x, f_x, f_n_x )

if ( __name__ == "__main__" ):
    main()
