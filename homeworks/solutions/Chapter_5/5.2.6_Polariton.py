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
    W0  = 1
    M   = 1
    X   = np.zeros( (NSTEPS) ) # One Particle
    V   = np.zeros( (NSTEPS) ) # One Particle

    # Photonic Parameters
    global QC0, VC0, WC, QC, VC, A0, do_DSE
    do_DSE = True
    A0     = 1.0
    QC0    = 0
    VC0    = 0
    WC     = 1
    QC     = np.zeros( (NSTEPS) ) # One Photon
    VC     = np.zeros( (NSTEPS) ) # One Photon

    global ENERGY
    ENERGY = np.zeros( (NSTEPS,3) )

    global DATA_DIR
    DATA_DIR = "5.2.6_Polariton/"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)


def get_X_FORCE( x, q ): # Electronic Force
    FORCE  = np.zeros( (1) ) # One Particle
    FORCE -= M * W0**2 * x + np.sqrt( 2 * WC**3 ) * A0 * q
    if ( do_DSE == True ):
        FORCE -= 2 * WC * A0**2 * x
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

    ENERGY[0,:] = get_Energy( X[0], V[0], QC[0], VC[0] )

    F0x = get_X_FORCE(  X[0], QC[0] )
    F0c = get_QC_FORCE( X[0], QC[0] )
    for step in range( NSTEPS-1 ):

        X[step+1]  = X[step]  + dt * V[step]  + 0.5 * dt**2 * F0x / M
        QC[step+1] = QC[step] + dt * VC[step] + 0.5 * dt**2 * F0c
        F1x        = get_X_FORCE(  X[step+1], QC[step+1] )
        F1c        = get_QC_FORCE( X[step+1], QC[step+1] )
        V[step+1]  = V[step]  + 0.5 * dt * (F0x + F1x) / M
        VC[step+1] = VC[step] + 0.5 * dt * (F0c + F1c)
        F0x        = F1x
        F0c        = F1c

        ENERGY[step+1,:] = get_Energy( X[step+1], V[step+1], QC[step+1], VC[step+1] )

def get_Energy( x,v,qc,vc ):

    E      = np.zeros( (3) )
    E[0]   = 0.5 * M * v**2  + 0.5 * vc**2 # Kinetic Energy
    E[1]   = 0.5 * M * W0**2 * x**2 + 0.5 * WC**2 * qc**2 # Bare Potential Energy
    E[1]  += np.sqrt(2 * WC**3) * A0 * x * qc # Interaction Potential Energy
    if ( do_DSE == True ):
        E[1] += WC * A0**2 * x**2 # Interaction Dipole Self-Energy (DSE)
    E[2]  = E[0] + E[1]
    return E

def plot():

    plt.plot( np.arange(NSTEPS)*dt, X[:], label="X" )
    plt.plot( np.arange(NSTEPS)*dt, QC[:], label="QC" )
    plt.legend()
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Position (a.u.)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Position_t.jpg", dpi=300)
    plt.clf()

    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,0], label="$E_{KIN}$" )
    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,1], label="$E_{POT}$" )
    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,2], label="$E_{TOT}$" )
    plt.legend()
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Energy_t.jpg", dpi=300)
    plt.clf()

    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,2] - ENERGY[0,2] )
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Total Energy (a.u.)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/Energy_t_Total.jpg", dpi=300)
    plt.clf()
    

def get_FFT():

    SMOOTH = np.sin( np.pi * np.arange(len(X))/len(X) ) # np.ones( len(X) )
    X_k = np.fft.fft( SMOOTH * X, n=2**16, norm='ortho' )
    #X_k = np.fft.fft( X, n=2**15, norm='ortho' )
    k   = np.fft.fftfreq( len(X_k) )

    X_k = np.roll( X_k, len(X_k)//2 )
    k   = np.roll( k, len(k)//2 )

    k   *= 2 * np.pi / dt

    return k, X_k   

def plot_FFT(k, X_k):

    plt.plot( k, np.abs(X_k), label="|X(k)|" )
    plt.legend()
    plt.xlabel("Frequency (a.u.)", fontsize=15)
    plt.ylabel("Position (a.u.)", fontsize=15)
    #RABI_FREQ = 2 * WC * A0
    #plt.xlim(np.max([0,np.average([WC,W0])-RABI_FREQ]),np.average([WC,W0])+RABI_FREQ)
    plt.xlim(0,4*WC)
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
