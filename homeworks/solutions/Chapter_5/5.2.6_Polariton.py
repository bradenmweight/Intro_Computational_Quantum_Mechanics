import numpy as np
from matplotlib import pyplot as plt

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
    global QC0, VC0, WC, QC, VC, A0
    A0  = 0.1
    QC0 = 0
    VC0 = 0
    WC  = 1
    QC  = np.zeros( (NSTEPS) ) # One Photon
    VC  = np.zeros( (NSTEPS) ) # One Photon

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
    plt.savefig("Position_t.jpg", dpi=300)


def main():
    get_Globals()
    propagate_VV()
    plot()

if ( __name__ == "__main__" ):
    main()
