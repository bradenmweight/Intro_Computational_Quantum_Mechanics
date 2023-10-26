import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

def get_Globals():
    # General Parameters
    global NSTEPS, dt, SIM_TIME 
    NSTEPS   = 10000
    dt       = 0.01
    SIM_TIME = NSTEPS * dt
    
    # Electronic Parameters
    global X0, V0, W0, X, V, M
    X0  = 1
    V0  = 0
    W0  = 2
    M   = 1
    X   = np.zeros( (NSTEPS) ) # One Particle
    V   = np.zeros( (NSTEPS) ) # One Particle

    # Photonic Parameters
    global QC0, VC0, WC, QC, VC, A0
    A0  = 1
    QC0 = 0
    VC0 = 0
    WC  = 0.5
    QC  = np.zeros( (NSTEPS) ) # One Photon
    VC  = np.zeros( (NSTEPS) ) # One Photon

    global DATA_DIR
    DATA_DIR = "5.2.6_Polariton/"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)


def get_X_FORCE( x, q ): # Electronic Force
    FORCE  = np.zeros( (1) ) # One Particle
    FORCE -= M * W0**2 * x + np.sqrt( 2 * WC**3 ) * A0 * q + 2 * WC * A0**2 * x
    return FORCE

def get_QC_FORCE( x, q ): # Photonic Force
    FORCE  = np.zeros( (1) ) # One Photon
    FORCE -= WC**2 * q + np.sqrt( 2 * WC**3 ) * A0 * x
    return FORCE

def propagate_VV():
    X[0]  = X0
    V[0]  = V0
    QC[0] = QC0
    VC[0] = VC0

    F0x = get_X_FORCE(  X[0], QC[0] )
    F0c = get_QC_FORCE( X[0], QC[0] )
    for step in range( NSTEPS-1 ):

        X[step+1]  = X[step]  + dt * V[step]  + 0.5 * dt**2 * F0x / M
        QC[step+1] = QC[step] + dt * VC[step] + 0.5 * dt**2 * F0c
        F1x        = get_X_FORCE( X[step+1], QC[step+1] )
        F1c        = get_QC_FORCE( X[step+1], QC[step+1] )
        V[step+1]  = V[step]  + 0.5 * dt * (F0x + F1x) / M
        VC[step+1] = VC[step] + 0.5 * dt * (F0c + F1c)
        F0x        = F1x
        F0c        = F1c

def plot():

    plt.plot( np.arange(NSTEPS), X[:], label="X" )
    plt.plot( np.arange(NSTEPS), QC[:], label="QC" )
    plt.legend()
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Position (a.u.)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Position_t.jpg", dpi=300)
    plt.clf()

def get_FFT():

    SMOOTH = np.sin( np.pi * np.arange(len(X))/len(X) ) # np.ones( len(X) )
    X_k = np.fft.fft( SMOOTH * X, n=2**15, norm='ortho' )
    #X_k = np.fft.fft( X, n=2**15, norm='ortho' )
    k   = np.fft.fftfreq( len(X_k) )

    X_k = np.roll( X_k, len(X_k)//2 )
    k   = np.roll( k, len(k)//2 )

    k *= 2 * np.pi / dt

    return k, X_k   

def plot_FFT(k, X_k):

    plt.plot( k, np.abs(X_k), label="|X_k|" )
    #plt.plot( k, X_k.real, label="Re[X_k]" )
    #plt.plot( k, X_k.imag, label="Im[X_k]" )
    plt.legend()
    plt.xlabel("Frequency (a.u.)", fontsize=15)
    plt.ylabel("Position (a.u.)", fontsize=15)
    plt.xlim(-5,5)
    plt.savefig(f"{DATA_DIR}/Position_w.jpg", dpi=300)
    plt.clf()

def main():
    get_Globals()
    propagate_VV()
    plot()
    k, X_k = get_FFT()
    plot_FFT( k, X_k )

if ( __name__ == "__main__" ):
    main()
