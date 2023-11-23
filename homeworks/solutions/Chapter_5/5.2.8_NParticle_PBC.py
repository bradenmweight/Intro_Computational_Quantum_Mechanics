import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from numba import jit

DATA_DIR = "5.2.8_NParticle_PBC"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global NSTEPS, dt, NPARTICLES, NSKIP
    NSTEPS     = 10000
    dt         = 0.1
    NPARTICLES = 1000
    NSKIP      = int(np.ceil(2/dt)) # Save every 2 a.u.t.

    global COORDS, VELOCS, MASSES, ENERGY, TEMP
    COORDS = np.zeros( (NSTEPS,NPARTICLES) )
    VELOCS = np.zeros( (NSTEPS,NPARTICLES) )
    MASSES = np.ones ( (NPARTICLES) )
    ENERGY = np.zeros( (NSTEPS,3) ) # EKIN, EPOT, ETOT
    TEMP   = np.zeros( (NSTEPS) )
    
    COORDS[0,:] = np.arange( 1,NPARTICLES+1 )*2
    VELOCS[0,:] = np.zeros( (NPARTICLES) )

    global L, EPS, SIG
    L = (2*NPARTICLES + 1) * 1.25
    EPS = 10.0
    SIG = 1.0


@jit(nopython=True)
def check_PBC( R ):
    for p in range( len(R) ):
        if ( R[p] < 0 ):
            R[p] = R[p] + L
        elif ( R[p] > L ):
            R[p] = R[p] - L
    return R

@jit(nopython=True)
def get_Force( R, step ):

    FORCE = np.zeros( (NPARTICLES) )
    for p in range( NPARTICLES ):
        for pp in range( p+1, NPARTICLES ):
            R12  = R[pp] - R[p]
            if ( abs(R12) > L/2 ):
                if ( R[pp] > L/2 ):
                    R12 = (R[pp] - L) - R[p]
                elif ( R[pp] < L/2 ):
                    R12 = (R[pp] + L) - R[p]
            R12_NORM   = abs(R12) # np.linalg.norm( R12 ) # |R1 - R2| = sqrt( dx^2 + dy^2 + dz^2 )
            R12_UNIT   = R12 / R12_NORM           
            
            FORCE[p]  += -1 * 4 * EPS * ( -12 * SIG**12 / R12_NORM**13 + 6 * SIG**6 / R12_NORM**7 ) * R12_UNIT
            FORCE[pp] += -1 * FORCE[p] # Equal and opposite force. Thanks, Newton.
           

    return FORCE


def propagate():

    ENERGY[0,:] = get_Energy( COORDS[0,:], VELOCS[0,:] )
    TEMP[0]     = get_Temperature( ENERGY[0,0], NPARTICLES )
    F0          = get_Force( COORDS[0], 0 )
    
    for step in range( NSTEPS-1 ):
        if ( step % 100 == 0 ):
            print("Step %1.0f of %1.0f" % (step, NSTEPS))

        # Do volecity-Verlet
        COORDS[step+1] = COORDS[step] + dt * VELOCS[step] + 0.5 * dt**2 * F0 / MASSES[:]
        F1 = get_Force( COORDS[step+1], step )
        VELOCS[step+1] = VELOCS[step] + 0.5 * dt * (F0 + F1) / MASSES[:]
        F0 = F1
        
        # Check the boundary conditions
        COORDS[step+1] = check_PBC( COORDS[step+1] )

        ENERGY[step+1,:] = get_Energy( COORDS[step+1,:], VELOCS[step+1,:] )
        TEMP[step+1]     = get_Temperature( ENERGY[step+1,0], NPARTICLES )

@jit(nopython=True)
def get_Energy( R, V ):
    E = np.zeros( (3) )
    for p in range( NPARTICLES ):
        E[0] += 0.5 * MASSES[p] * V[p]**2 # 0.5 * np.linalg.norm(MASSES[p] * V[p])**2
        for pp in range( p+1,NPARTICLES ):
            R12        = R[pp] - R[p]
            if ( abs(R12) > L/2 ):
                if ( R[pp] > L/2 ):
                    R12 = (R[pp] - L) - R[p]
                elif ( R[pp] < L/2 ):
                    R12 = (R[pp] + L) - R[p]
            R12_NORM   = abs(R12) # np.linalg.norm( R12 ) # |R1 - R2| = sqrt( dx^2 + dy^2 + dz^2 )
            E[1]      -= 4 * EPS * ( SIG**12 / R12_NORM**12 - SIG**6 / R12_NORM**6 )
    
    E[2] = E[0] + E[1]
    return E

@jit(nopython=True)
def get_Temperature( EK, NPARTICLES ):
    EK *= 27.2114
    KT  = 300 / 0.025
    T   = 2 * EK / NPARTICLES / KT # / 3 # Divide by 3 for 3D
    return T


def plot():
    plt.plot( np.arange(NSTEPS)*dt, COORDS[:], "o" )
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Position (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/X.jpg", dpi=300)
    plt.clf()

    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,0], "-", label="KIN" )
    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,1], "-", label="POT" )
    plt.plot( np.arange(NSTEPS)*dt, ENERGY[:,2], "-", label="TOT" )
    plt.legend()
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/E.jpg", dpi=300)
    plt.tight_layout()
    plt.clf()

    plt.plot( np.arange(NSTEPS)*dt, TEMP[:], "-" )
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Temperature (K)", fontsize=15)
    plt.savefig(f"{DATA_DIR}/T.jpg", dpi=300)
    plt.tight_layout()
    plt.clf()


def save_XYZ():

    # Create XYZ File
    FILE01 = open(f"{DATA_DIR}/Trajectory.xyz","w")
    for step in range( 0, NSTEPS, NSKIP ):
        FILE01.write(f"{NPARTICLES}\n")
        FILE01.write("MD Step: %1.0f   Time: %1.4f\n" %(step, step*dt))
        for p in range( NPARTICLES ):
            FILE01.write( "X %1.4f %1.4f %1.4f\n" % (COORDS[step,p],COORDS[step,p],COORDS[step,p]) )
    FILE01.close()

def main():
    get_Globals()
    propagate()
    plot()
    save_XYZ()

if ( __name__ == "__main__" ):
    main()




