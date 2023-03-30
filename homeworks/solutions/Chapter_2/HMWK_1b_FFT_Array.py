import numpy as np
import subprocess as sp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Make a data folder
DATA_DIR = "PLOTS_HMWK_1"
sp.call(f"mkdir {DATA_DIR}", shell=True)

def get_4x4_DFT(): # 0, 1, 2, 3, ...
    """
    Returns Manually Created 4x4 Discrete Fourier Transform (DFT) Matrix.
    """    
    gamma = np.exp(-2*np.pi/4 * 1j)
    W = np.zeros(( 4,4 ), dtype=complex)
    W[:,0] = 1 + 0j
    W[0,:] = 1 + 0j
    
    W[1,1] = gamma

    W[1,2] = gamma ** 2
    W[2,1] = gamma ** 2
    
    W[1,3] = gamma ** 3
    W[3,1] = gamma ** 3
    
    W[2,2] = gamma ** (2*2)
    
    W[2,3] = gamma ** (2*3)
    W[3,2] = gamma ** (3*2)
    
    W[3,3] = gamma ** (3*3)

    #print("4x4 Manual DFT Matrix:\n",W)
    np.savetxt(f"{DATA_DIR}/W_MANUAL_4x4.dat", W)


def get_NxN_DFT(N): # 0, 1, 2, 3, ...
    """
    Returns Nth-dimensional Discrete Fourier Transform (DFT) Matrix.
    """    
    gamma = np.exp(-2*np.pi/N * 1j)
    W = np.zeros(( N,N ), dtype=complex)
    for j in range(N):
        for k in range(N):
            W[j,k] = gamma ** (j*k)
    print(f"{N}x{N} Automatic DFT Matrix:\n",W)
    print(W)
    np.savetxt(f"{DATA_DIR}/W_AUTO_{N}x{N}.dat", W)
    return W

def save_plot(W,N):

    # Save real part of matrix
    plt.imshow(np.real(W),origin='lower',cmap="seismic")
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/W_AUTO_{N}x{N}_RE.jpg",dpi=600)
    plt.clf()

    # Save imaginary part of matrix
    plt.imshow(np.imag(W),origin='lower',cmap="seismic")
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/W_AUTO_{N}x{N}_IM.jpg",dpi=600)
    plt.clf()

    # Save absolute value of matrix
    plt.imshow(np.abs(W),origin='lower',cmap="seismic")
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/W_AUTO_{N}x{N}_ABS.jpg",dpi=600)
    plt.clf()

def main():

    N = 100 # Any number can go here.
    get_4x4_DFT()
    W = get_NxN_DFT(N)
    save_plot(W,N)

if ( __name__ == "__main__" ):
    main()