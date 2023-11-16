import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "5.2.7_2Particle_PBC"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global NSTEPS, dt
    NSTEPS = 100000
    dt = 0.01

    global COORDS, VELOCS, MASSES
    COORDS = np.zeros( (NSTEPS,2) )
    VELOCS = np.zeros( (NSTEPS,2) )
    MASSES = np.array([1,1])
    
    COORDS[0,:] = np.array([1.0,5.0])
    VELOCS[0,:] = np.array([0.1,0.0])

    global L
    L = 10.0

    global do_PBC
    do_PBC = True

def check_PBC( R ):
    if ( do_PBC == True ):
        for p in range( len(R) ):
            if ( R[p] < 0 ):
                R[p] = R[p] + L
            elif ( R[p] > L ):
                R[p] = R[p] - L
    return R

def get_Force( R, step ):

    EPS = 10.0
    SIG = 0.2
    N   = 2
    
    FORCE = np.zeros( (N) )

    for p in range( N ):
        for pp in range( p+1, N ):
            R12  = R[p] - R[pp]
            if ( do_PBC == True ):
                if ( abs(R12) > L/2 ):
                    if ( R[p] > L/2 ):
                        R12 = (R[p] - L) - R[pp]
                    elif ( R[p] < L/2 ):
                        R12 = R[p] - (R[pp] - L)

            R12 *= -1
            R12_NORM   = np.linalg.norm( R12 ) # |R1 - R2| = sqrt( dx^2 + dy^2 + dz^2 )
            R12_UNIT   = R12 / R12_NORM           
            
            FORCE[p]  += -1 * 4 * EPS * ( -12 * SIG**12 / R12_NORM**13 + 6 * SIG**6 / R12_NORM**7 ) * R12_UNIT
            FORCE[pp] += -1 * FORCE[p] # Equal and opposite force. Thanks, Newton.
           

    return FORCE


def propagate():
    F0 = get_Force( COORDS[0], 0 )
    
    for step in range( NSTEPS-1 ):

        # Do volecity-Verlet
        COORDS[step+1] = COORDS[step] + dt * VELOCS[step] + 0.5 * dt**2 * F0 / MASSES[:]
        F1 = get_Force( COORDS[step+1], step )
        VELOCS[step+1] = VELOCS[step] + 0.5 * dt * (F0 + F1) / MASSES[:]
        F0 = F1
        
        # Check the boundary conditions
        COORDS[step+1] = check_PBC( COORDS[step+1] )


def plot():
    plt.plot( np.arange(NSTEPS)*dt, COORDS[:], "o" )
    plt.savefig(f"{DATA_DIR}/X.jpg", dpi=300)

def main():
    get_Globals()
    propagate()
    plot()

if ( __name__ == "__main__" ):
    main()




