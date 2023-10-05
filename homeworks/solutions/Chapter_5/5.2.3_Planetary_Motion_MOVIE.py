import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from numba import jit
from time import time

import imageio.v2 as imageio
from pygifsicle import optimize as gifOPT # This needs to be installed somewhere
from PIL import Image, ImageDraw, ImageFont

"""
Install pygifcicle:
pip3 install pygifsicle


Install gifsicle: ( echo "$(pwd)" = /scratch/bweight/software/ )
curl -sL http://www.lcdf.org/gifsicle/gifsicle-1.91.tar.gz | tar -zx
cd gifsicle-1.91
./configure --disable-gifview
make install exec_prefix=$(pwd) prefix=$(pwd) datarootdir=$(pwd)
"""

def get_Globals():
    global NOBJECTS, NSTEPS, dt
    global INIT_POS, INIT_VEL, MASSES
    NOBJECTS = 2
    dt       = 0.5
    NSTEPS   = 1000
    SIM_TIME = NSTEPS * dt
    print("\tTotal Simulation Time: %1.0f a.u." % SIM_TIME)
    INIT_POS = np.zeros( (NOBJECTS,3) )
    INIT_VEL = np.zeros( (NOBJECTS,3) )
    MASSES   = np.zeros( (NOBJECTS) )
    
    #### Circular Orbit ####
    #### NOBJECTS = 2
    # INIT_POS[0,:] = np.array( [0,0,0] )
    # INIT_POS[1,:] = np.array( [1000,0,0] )
    # INIT_VEL[0,:] = np.array( [0,0,0] )
    # MASSES[0]     = 100000
    # MASSES[1]     = 10
    # R12           = np.linalg.norm(INIT_POS[0,:] - INIT_POS[1,:])
    # INIT_VEL[1,1] = np.sqrt((MASSES[0] + MASSES[1]) / R12) # Y-direction -- Perp. to \hat{r}
    ########################

    #### Elliptical Orbit ####
    #### NOBJECTS = 2
    INIT_POS[0,:] = np.array( [0,0,0] )
    INIT_POS[1,:] = np.array( [1000,0,0] )
    INIT_VEL[0,:] = np.array( [0,0,0] )
    MASSES[0]     = 100000
    MASSES[1]     = 10
    R12           = np.linalg.norm(INIT_POS[0,:] - INIT_POS[1,:])
    INIT_VEL[1,1] = np.sqrt((MASSES[0] + MASSES[1]) / R12)/2 # Y-direction -- Perp. to \hat{r}
    ########################

    #### Escapting Object ####
    #### NOBJECTS = 2
    # INIT_POS[0,:] = np.array( [0,0,0] )
    # INIT_POS[1,:] = np.array( [1000,0,0] )
    # INIT_VEL[0,:] = np.array( [0,0,0] )
    # MASSES[0]     = 100000
    # MASSES[1]     = 10
    # R12           = np.linalg.norm(INIT_POS[0,:] - INIT_POS[1,:])
    # INIT_VEL[1,1] = np.sqrt((MASSES[0] + MASSES[1]) / R12)*np.sqrt(2) # Y-direction -- Perp. to \hat{r}
    ########################


    #### 3-Body Orbits (Unstable) ####
    # INIT_POS[0,:] = np.array( [-1000,0,0] )
    # INIT_POS[1,:] = np.array( [0,1000,0] )
    # INIT_POS[2,:] = np.array( [1000,0,0] )
    # INIT_VEL[0,:] = np.array( [0,-1,0] )
    # INIT_VEL[1,:] = np.array( [0,0,0] )
    # INIT_VEL[2,:] = np.array( [0,1,0] )
    # MASSES[0]     = 10000
    # MASSES[1]     = 100
    # MASSES[2]     = 10000
    ########################

    global COORDS, VELOCS
    COORDS = np.zeros( (NOBJECTS,NSTEPS,3) )
    VELOCS = np.zeros( (NOBJECTS,NSTEPS,3) )

    global ENERGY
    ENERGY = np.zeros( (3,NSTEPS) ) # EKIN, EPOT, ETOT

    global DATA_DIR
    DATA_DIR = "5.2.3_Planetary_Motion/"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

@jit(nopython=True)
def get_Gravitational_Force( R ):
    """
    Gravitational Potential: V_G = mM / |R1 - R2|
    Gravitational Force:     
        F_g           = -1 * \\nabla V_G = -1 * mM / |R1 - R2|^2 * \hat{R1 - R2}
        |R1 - R2|     = sqrt( dx^2 + dy^2 + dz^2 )
        \hat{R1 - R2} = (R1 - R2) / |R1 - R2|
    """
    FORCE = np.zeros( (NOBJECTS,3) )
    for p in range( NOBJECTS ):
        for pp in range( p+1,NOBJECTS ):
            R12        = R[p,:] - R[pp,:]
            R12_NORM   = np.linalg.norm( R12 )
            R12_UNIT   = R12 / R12_NORM           
            
            FORCE[p,:]  += -1 * MASSES[p] * MASSES[pp] / R12_NORM**2 * R12_UNIT
            FORCE[pp,:] += -1 * FORCE[p,:] # Equal and opposite force. Thanks, Newton.
    return FORCE

def propagate_VV():

    COORDS[:,0,:] = INIT_POS
    VELOCS[:,0,:] = INIT_VEL
    ENERGY[:,0]   = get_Energy( COORDS[:,0,:], VELOCS[:,0,:] )

    T1 = time()
    F0 = get_Gravitational_Force( COORDS[:,0,:] )
    for step in range( NSTEPS-1 ):

        # Verlet Approach 1
        #COORDS[:,step+1,:]  = COORDS[:,step,:] + 0.500 * dt * VELOCS[:,step,:]
        #VELOCS[:,step+1,:]  = VELOCS[:,step,:] + dt * get_Gravitational_Force( COORDS[:,step+1,:] ) / MASSES[:,None]
        #COORDS[:,step+1,:] += 0.500 * dt * VELOCS[:,step+1,:]

        # Verlet Approach 2 -- Most often used
        COORDS[:,step+1,:] = COORDS[:,step,:] + dt * VELOCS[:,step,:] + 0.500 * dt**2 * F0 / MASSES[:,None]
        F1                 = get_Gravitational_Force( COORDS[:,step+1,:] )
        VELOCS[:,step+1,:] = VELOCS[:,step,:] + 0.500 * dt * ( F0 + F1 ) / MASSES[:,None]
        F0 = F1

        ENERGY[:,step+1] = get_Energy( COORDS[:,step+1,:], VELOCS[:,step+1,:] )
        
    print("\tTotal CPU Time: %1.3f seconds" % (time() - T1) )

@jit(nopython=True)
def get_Energy( R, V ):
    E = np.zeros( (3) )
    for p in range( NOBJECTS ):
        E[0] += 0.5 * MASSES[p] * np.linalg.norm(V[p,:])**2
        for pp in range( p+1,NOBJECTS ):
            E[1] -= MASSES[p] * MASSES[pp] / np.linalg.norm(R[p,:] - R[pp,:])
    
    E[2] = E[0] + E[1]
    return E

def plot():

    ##### 3D Figure #####
    #ax = plt.figure().add_subplot(projection='3d')
    #for p in range( NOBJECTS ):
    #    ax.plot( COORDS[p,:,0], COORDS[p,:,1], COORDS[p,:,2], c='black', label=f"O{p}" )
    
    for p in range( NOBJECTS ):
        plt.plot( COORDS[p,:,0], COORDS[p,:,1], label=f"Obj. {p}" )
    plt.legend()
    plt.xlabel("Position X (a.u.)",fontsize=15)
    plt.ylabel("Position Y (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/Trajectory.jpg",dpi=300)
    plt.clf()

    for p in range( NOBJECTS ):
        plt.plot( VELOCS[p,:,0], VELOCS[p,:,1], label=f"Obj. {p}" )
    plt.legend()
    plt.xlabel("Velocity X (a.u.)",fontsize=15)
    plt.ylabel("Velocity Y (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/Velocity.jpg",dpi=300)
    plt.clf()

    plt.plot( np.arange(NSTEPS)*dt, ENERGY[0,:], c='black', label="$E_\mathrm{Kin}$" )
    plt.plot( np.arange(NSTEPS)*dt, ENERGY[1,:], c='red', label="$E_\mathrm{Pot}$" )
    plt.plot( np.arange(NSTEPS)*dt, ENERGY[2,:], c='green', label="$E_\mathrm{Tot}$" )
    plt.legend()
    plt.xlabel("Time (a.u.)",fontsize=15)
    plt.ylabel("Energy (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/Energy.jpg",dpi=300)
    plt.clf()

    plt.plot( np.arange(NSTEPS)*dt, ENERGY[2,:] - ENERGY[2,0], c='black', label="$E_\mathrm{Tot}$" )
    plt.legend()
    plt.xlabel("Time (a.u.)",fontsize=15)
    plt.ylabel("Energy (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/Energy_Total.jpg",dpi=300)
    plt.clf()

def make_movie():

    def make_frame( frame ):
        for p in range( NOBJECTS ):
            plt.plot( COORDS[p,:frame,0], COORDS[p,:frame,1] )
            if ( frame == 0 ):
                plt.scatter( COORDS[p,frame,0], COORDS[p,frame,1], label=f"Obj. {p}" )
            else:
                plt.scatter( COORDS[p,frame,0], COORDS[p,frame,1] )
        if ( frame == 0 ):
            plt.legend()
        plt.xlim( -1500, 1500 )
        plt.ylim( -1500, 1500 )
        plt.xlabel("Position X (a.u.)",fontsize=15)
        plt.ylabel("Position Y (a.u.)",fontsize=15)
        plt.title("Simulation Time: %1.2f" % (dt * frame),fontsize=15)
        plt.tight_layout()
        plt.savefig(f"DUMMY.jpg",dpi=150)
        plt.clf()


    movieNAME = f"{DATA_DIR}/Trajectory.gif"
    NSKIP     = 10
    with imageio.get_writer(movieNAME, mode='I', fps=5) as writer: # Get a writer object
        for frame in range( 0, NSTEPS, NSKIP ):
            print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1)//NSKIP, NSTEPS//NSKIP) )
            make_frame( frame )
            image = imageio.imread( "DUMMY.jpg" ) # Read JPEG file
            writer.append_data(image) # Write JPEG file (to memory at first; then printed at end)
    sp.call("rm DUMMY.jpg", shell=True)
    gifOPT(movieNAME) # This will compress the GIF movie by at least a factor of two/three. With this: ~750 frames --> 80 MB




def main():
    get_Globals()
    propagate_VV()
    plot()
    make_movie()

if ( __name__ == "__main__" ):
    main()