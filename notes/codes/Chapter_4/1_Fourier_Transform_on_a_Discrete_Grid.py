import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import subprocess as sp

from numpy import fft


def getGlobals():
    global DATA_DIR
    DATA_DIR = "1_PLOTS_DATA/"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_f_x():
    global x, dx, Lx, Nx

    # Gaussian Function
    x   = np.linspace(-20,20,1001)
    f_x = np.sqrt(1/2/np.pi) * np.exp(-x**2 / 2) + 0j

    ## Shifted Gaussian Function
    ## Shift in real-space results is phase change of the DFT
    ## f(x) --> f(x+dx) => f(k) --> f(k) * e^{-i k dx}
    ## This is called the "Fourier Shift Theorem"
    #x     = np.linspace(-20,20,1001)
    #global shift # Use this to plot analytic result later
    #shift = 1.0
    #f_x   = np.sqrt(1/2/np.pi) * np.exp(-(x-shift)**2 / 2) + 0j

    ## Gaussian Function with complex factor (opposite as previous one)
    ## Shift in real-space results is phase change of the DFT
    #x     = np.linspace(-20,20,1001)
    #global p0 # Use this to plot analytic result later
    #p0 = 1.0 # Initial momentum --> Intepret as a coordinate shift in momentum space
    #factor = np.exp( -1j * 2 * np.pi * p0 * x )
    #f_x   = np.sqrt(1/2/np.pi) * np.exp(-x**2 / 2) * factor

    ## Sinusoidal function of frequency k = +- 1
    ## Large grid sampling
    #x   = np.linspace(-20,20,1001)
    #f_x = np.sin( 2 * np.pi * x ) + 0j
    
    ## Sinusoidal function of frequency k = +- 1
    ## Small grid sampling
    #x   = np.linspace(-2,2,101)
    #f_x = np.sin( 2 * np.pi * x ) + 0j

    ## Addition of two waves
    #x   = np.linspace(-10,10,2001)
    #f_x = np.sin( 2 * np.pi * x ) + 0.5 * np.sin( np.pi * x ) + 0j
    
    ## Gaussian-dressed wave-packet
    #x   = np.linspace(-10,10,1001)
    #f_x = np.exp(-x**2 / 2 / 1**2) * np.sin( 2 * np.pi * x ) + 0j
    
    ##f(x) = 1.0 --> f(k) = delta(k)
    #x   = np.linspace(-10,10,1001)
    #f_x = np.ones( len(x) )
    
    ## Single-frequency plane-waves (only positive / only negative)
    #x   = np.linspace(-10,10,2001)
    #f_x = np.exp( 1j * 2 * np.pi * x ) # cos + sin
    #f_x = np.exp( -1j * 2 * np.pi * x ) # cos - sin

    ## Weird Function -- but which has a frequency
    #x   = np.linspace(-50,50,1001)
    #f_x = np.exp( np.sin( np.pi * x ) ) * np.exp( - x**2 / 2 / 5 )


    # Initialize useful numbers
    Nx  = len(x)
    Lx  = x[-1] - x[0] # Last minus first
    dx  = x[1]  - x[0] # Second minus first

    return f_x

def plot_f_x(f_x):
    plt.plot( x, f_x.real, c="black", label="RE" )
    plt.plot( x, f_x.imag, c="red", label="IM" )
    plt.legend()
    plt.xlim(x[0],x[-1])
    #plt.xlim(-2,2)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("f(x)", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/f_x.jpg")
    plt.clf()

def get_FT_oneSIDED( f_x ):

    # Define the k-grid (reciprocal grid)
    dk   = 2 * np.pi / Lx
    kmax = 1 / 2 / dx # This is not angular frequency
    k    = np.linspace( -kmax, kmax, Nx )

    # Define the Fourier matrix, W
    n = np.arange(Nx).reshape( (-1,1) )
    m = np.arange(Nx).reshape( (1,-1) )

    ### One-sided transform
    W = np.exp( -2j*np.pi * m * n / Nx )
    
    # Operate W on the real-space function
    f_k = W @ f_x
    
    ### Roll coordinates such that zero is centered
    ### Required for the one-sided transform
    f_k = np.roll( f_k, Nx//2 )

    # Add normalization and integral infinitesimal
    f_k *= dx / np.sqrt( 2 * np.pi )

    return k, f_k


def get_FT_Centered( f_x ):

    # Define the k-grid (reciprocal grid)
    dk   = 2 * np.pi / Lx
    kmax = 1 / 2 / dx # This is not angular frequency
    k    = np.linspace( -kmax, kmax, Nx )

    # Define the Fourier matrix, W
    n = np.arange(Nx).reshape( (-1,1) )
    m = np.arange(Nx).reshape( (1,-1) )

    ### Centered transform
    ### Here, we shift the indices to the center
    a = (Nx)//2
    W = np.exp( -2j*np.pi * (m-a) * (n-a) / Nx )
    
    # Operate W on the real-space function
    f_k = W @ f_x
    
    # Add normalization and integral infinitesimal
    f_k *= dx / np.sqrt( 2 * np.pi )

    return k, f_k

def get_numpy_FT( f_x ):
    # Get the fourier transform
    k   = fft.fftfreq( len(f_x), d=dx )
    f_k = fft.fft( f_x, norm="ortho" ) * np.sqrt( np.pi / 2 )

    # Shift so that most negative k is first
    f_k = np.roll( f_k, Nx//2 )
    k    = np.roll( k, Nx//2 )

    return k, f_k

def plot_f_k(k,f_k,k_np,f_k_np,title):

    # Plot a version comparing directly to Numpy
    plt.plot( k, np.abs(f_k.real), "-", c='black', lw=10, alpha=0.5, label="RE (Manual)" )
    plt.plot( k, np.abs(f_k.imag), "o-", c='black', lw=10, alpha=0.5, label="IM (Manual)" )

    plt.plot( k_np,np.abs(f_k_np.real),"-", c='red', lw=2, label="RE (Numpy)" )
    plt.plot( k_np,np.abs(f_k_np.imag), "o-", c='red', lw=2, label="IM (Numpy)" )

    # Gaussian function in k-space -- Analytic Result
    #plt.plot( k, np.sqrt(1/2/np.pi) * np.exp(-2 * k**2 * np.pi**2), c="green", label="Gaussian" )

    plt.legend()
    plt.xlim(-2,2 )
    plt.xlabel("k", fontsize=18)
    plt.ylabel("f(k)", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/f_k_{title}.jpg")
    plt.clf()


    #########################################
    # Plot a ''clean'' one without comparing to Numpy
    plt.plot( k, f_k.real, "-", c='black', lw=2, label="RE" )
    plt.plot( k, f_k.imag, "-", c='red',   lw=2, label="IM" )

    # Gaussian function in k-space -- Analytic Result
    #analytic_func_k = np.sqrt(1/2/np.pi) * np.exp(-2 * k**2 * np.pi**2)
    #plt.plot( k, analytic_func_k, c="green", label="Gaussian" )

    # Shifted Gaussian function
    #analytic_func_k = np.sqrt(1/2/np.pi) * np.exp(-2 * k**2 * np.pi**2)
    #shifted_analytic = np.exp(-1j * 2 * np.pi * k * shift)  * analytic_func_k
    #plt.plot( k, shifted_analytic.real, "o", c="black", label="Analytic RE" )
    #plt.plot( k, shifted_analytic.imag, "o", c="red", label="Analytic IM" )

    plt.legend()
    plt.xlim(-2,2 )
    plt.xlabel("k", fontsize=18)
    plt.ylabel("f(k)", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/f_k_{title}_clean.jpg")
    plt.clf()





def main():

    getGlobals()
    f_x = get_f_x()
    plot_f_x(f_x)
    k, f_k = get_FT_oneSIDED(f_x)
    k_np, f_k_np = get_numpy_FT(f_x)
    plot_f_k(k, f_k, k_np, f_k_np,title="One-Sided")
    k, f_k = get_FT_Centered(f_x)
    plot_f_k(k, f_k, k_np, f_k_np,title="Centered")



if ( __name__ == "__main__" ):
    main()


