import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "5.2.6_1Particle_PBC"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global NSTEPS, dt
    NSTEPS = 10000
    dt = 0.1

    global COORDS, VELOCS, MASSES
    COORDS = np.zeros( (NSTEPS) )
    VELOCS = np.zeros( (NSTEPS) )
    MASSES = np.array([1])
    
    COORDS[0] = np.array([5.0])
    VELOCS[0] = np.array([0.1])

    global L
    L = 10.0

    global do_PBC
    do_PBC = False

def check_PBC( R ):
    if ( do_PBC == True ):
        if ( R < 0 ):
            R = R + L
        elif ( R > L ):
            R = R - L
    return R

def propagate():
    for step in range( NSTEPS-1 ):
        COORDS[step+1] = COORDS[step] + dt * VELOCS[step] # No forces
        COORDS[step+1] = check_PBC( COORDS[step+1] )
        VELOCS[step+1] = VELOCS[step] # No forces

def plot():
    plt.plot( np.arange(NSTEPS)*dt, COORDS[:] )
    plt.savefig(f"{DATA_DIR}/X.jpg", dpi=300)

def main():
    get_Globals()
    propagate()
    plot()

if ( __name__ == "__main__" ):
    main()




