import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import subprocess as sp
from numba import jit
from time import time
from random import random

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

Add to "~/.bashrc":
export PATH="${HOME}/gifsicle-1.91/bin:$PATH"
"""


def get_Globals():
    global NPARTICLES, NSTEPS, dt
    global MASSES, NFRAMES, NXYZ
    global INITIALIZE, DAMP, do_Check_Boundary
    NPARTICLES = 9
    INITIALIZE = "2D" # "1D", "2D", "3D"
    do_Check_Boundary = True # Confine dynamics to a box
    DAMP       = 0.0
    dt         = 1e-2
    NSTEPS     = 500000
    SIM_TIME   = NSTEPS * dt
    MASSES     = np.zeros( (NPARTICLES,3) ) # Make same shape as force
    NFRAMES    = 1000 # Only use this many frames for the movie
    NXYZ       = 1000 # Save this many XYZ coordinates

    global EPS, SIG
    EPS = 1.0
    SIG = 1.0

    global COORDS, VELOCS, ENERGY
    COORDS = np.zeros( (NPARTICLES,NSTEPS,3) )
    VELOCS = np.zeros( (NPARTICLES,NSTEPS,3) )
    ENERGY = np.zeros( (3,NSTEPS) ) # EKIN, EPOT, ETOT

    global DATA_DIR
    DATA_DIR = "5.2.4_Lennard-Jones_Fluid/"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

    print("\tTotal Simulation Time: %1.4f a.u." % SIM_TIME)

def get_Initial_COORDS_VELOCS():

    global BOX_SIZE

    # Initialize Masses
    for p in range( NPARTICLES ): # 0,1,2,3,N-1
        MASSES[p,:] = 1.0 * np.ones( (3) )


    if ( INITIALIZE == "1D" ):
        # Initialize Positions in SC Square Lattice
        a0  = SIG * 2**(1/6) #* 0.8
        V   = np.array([[1,0,0]]) * a0
        for xi in range( NPARTICLES ):
            C = V[0,:] * xi + (random()*2-1)/50
            COORDS[ xi,0,: ] = C
        BOX_SIZE = [-NPARTICLES * a0 / 2, NPARTICLES * a0 + 1 ]

    elif ( INITIALIZE == "2D" ):
        # Initialize Positions in SC Square Lattice
        a0  = SIG * 2**(1/6) #* 0.8
        V   = np.array([[1,0,0],[0,1,0]]) * a0
        N12 = int( NPARTICLES ** (1/2) )
        count = 0
        for xi in range( N12 ):
            for yi in range( N12 ):
                C = V[0,:] * xi + V[1,:] * yi
                COORDS[ count,0,: ] = C
                count += 1
        BOX_SIZE = [-N12 * a0/2, N12 * a0 + 1 ]
        assert( NPARTICLES - N12**2 == 0 ),"Number of particles should be a square: 4, 9, 16, etc."

    elif ( INITIALIZE == "3D" ):
        # Initialize Positions in SC Cubic Lattice
        a0  = SIG * 0.8
        V   = np.array([[1,0,0],[0,1,0],[0,0,1]]) * a0
        N13 = int( NPARTICLES ** (1/3) )
        count = 0
        for xi in range( N13 ):
            for yi in range( N13 ):
                for zi in range( N13 ):
                    C = V[0,:] * xi + V[1,:] * yi + V[2,:] * zi
                    COORDS[ count,0,: ] = C
                    count += 1
        BOX_SIZE = [-N13 * a0/2, N13 * a0 + 1 ]
        assert( NPARTICLES - N13**3 == 0 ),"Number of particles should be a cube: 8, 27, etc."







@jit(nopython=True)
def get_Force( R, V, SIG, EPS ):
    """
    Lennard-Jones Potential: V_LJ = 4*eps ( sig^12 / |R1 - R2|^12 - sig^6 / |R1 - R2|^6 )
    Lennard-Jones Force:     
        F_g           = -1 * \\nabla V_LG = -1 * 4*eps ( -12 * sig^12 / |R1 - R2|^13 + 6 * sig^6 / |R1 - R2|^7 ) \hat{R1 - R2}
        |R1 - R2|     = sqrt( dx^2 + dy^2 + dz^2 )
        \hat{R1 - R2} = (R1 - R2) / |R1 - R2|
    """
    FORCE = np.zeros( (NPARTICLES,3) )
    for p in range( NPARTICLES ):
        for pp in range( p+1,NPARTICLES ):
            R12        = R[p,:] - R[pp,:]
            R12_NORM   = np.linalg.norm( R12 ) # |R1 - R2| = sqrt( dx^2 + dy^2 + dz^2 )
            R12_UNIT   = R12 / R12_NORM           
            
            FORCE[p,:]  += -1 * 4 * EPS * ( -12 * SIG**12 / R12_NORM**13 + 6 * SIG**6 / R12_NORM**7 ) * R12_UNIT
            FORCE[pp,:] += -1 * FORCE[p,:] # Equal and opposite force. Thanks, Newton.

        # Damping of the velocities (i.e., frictional force -- solvent)
        VNORM       = np.sqrt( np.sum(V[p,:]**2) )
        VUNIT       = V[p,:] / VNORM
        FDAMP       = -1 * DAMP * VNORM**2 * VUNIT
        #FORCE[p,:] += FDAMP

    return FORCE

def check_boundary( R, V ):
    if ( do_Check_Boundary == True ):
        for p in range( NPARTICLES ):
            for d in range( 3 ):
                if ( R[p,d] < BOX_SIZE[0] or R[p,d] > BOX_SIZE[1] ):
                    V[p,d] *= -1
    return V

def propagate_VV():

    ENERGY[:,0] = get_Energy( COORDS[:,0,:], VELOCS[:,0,:] )

    F0 = get_Force( COORDS[:,0,:], VELOCS[:,0,:], SIG, EPS )
    for step in range( NSTEPS-1 ):
        if ( step%1000 == 0 ):
            print (f"Step {step} of {NSTEPS}")

        COORDS[:,step+1,:] = COORDS[:,step,:] + dt * VELOCS[:,step,:] + 0.500 * dt**2 * F0 / MASSES
        F1                 = get_Force( COORDS[:,step+1,:], VELOCS[:,step,:], SIG, EPS )
        VELOCS[:,step+1,:] = VELOCS[:,step,:] + 0.500 * dt * ( F0 + F1 ) / MASSES
        check_boundary( COORDS[:,step+1,:], VELOCS[:,step+1,:] )

        F0 = F1

        ENERGY[:,step+1] = get_Energy( COORDS[:,step+1,:], VELOCS[:,step+1,:] )
        


@jit(nopython=True)
def get_Energy( R, V ):
    E = np.zeros( (3) )
    for p in range( NPARTICLES ):
        E[0] += 0.5 * np.linalg.norm(MASSES[p,:] * V[p,:])**2
        for pp in range( p+1,NPARTICLES ):
            R12        = R[p,:] - R[pp,:]
            R12_NORM   = np.linalg.norm( R12 ) # |R1 - R2| = sqrt( dx^2 + dy^2 + dz^2 )
            R12_UNIT   = R12 / R12_NORM     
            E[1]      += 4 * EPS * ( SIG**12 / R12_NORM**12 - SIG**6 / R12_NORM**6 )
    
    E[2] = E[0] + E[1]
    return E

def plot():

    ##### 3D Figure #####
    #ax = plt.figure().add_subplot(projection='3d')
    #for p in range( NPARTICLES ):
    #    ax.plot( COORDS[p,:,0], COORDS[p,:,1], COORDS[p,:,2], c='black', label=f"O{p}" )
    
    # for p in range( NPARTICLES ):
    #     plt.plot( COORDS[p,:,0], COORDS[p,:,1], label=f"Obj. {p}" )
    # plt.legend()
    # plt.xlabel("Position X (a.u.)",fontsize=15)
    # plt.ylabel("Position Y (a.u.)",fontsize=15)
    # plt.savefig(f"{DATA_DIR}/Trajectory.jpg",dpi=300)
    # plt.clf()

    # for p in range( NPARTICLES ):
    #     plt.plot( VELOCS[p,:,0], VELOCS[p,:,1], label=f"Obj. {p}" )
    # plt.legend()
    # plt.xlabel("Velocity X (a.u.)",fontsize=15)
    # plt.ylabel("Velocity Y (a.u.)",fontsize=15)
    # plt.savefig(f"{DATA_DIR}/Velocity.jpg",dpi=300)
    # plt.clf()

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
        for p in range( NPARTICLES ):
            plt.plot( COORDS[p,:frame,0], COORDS[p,:frame,1], alpha=0.5 )
            plt.scatter( COORDS[p,frame,0], COORDS[p,frame,1], c="black" )
        
        L = BOX_SIZE[1] - BOX_SIZE[0]
        box = patches.Rectangle( (BOX_SIZE[0], BOX_SIZE[0]), L, L, linewidth=6, edgecolor="black", fill=False )
        plt.gca().add_patch(box)

        #plt.xlim( np.min(COORDS[:,0,0])*1.5, np.max(COORDS[:,0,0])*1.5 )
        #plt.ylim( np.min(COORDS[:,0,1])*1.5, np.max(COORDS[:,0,1])*1.5 )
        plt.xlim( BOX_SIZE[0]-1, BOX_SIZE[1]+1 )
        plt.ylim( BOX_SIZE[0]-1, BOX_SIZE[1]+1 )
        plt.xlabel("Position X (a.u.)",fontsize=15)
        plt.ylabel("Position Y (a.u.)",fontsize=15)
        plt.title("Simulation Time: %1.1f a.u." % (dt * frame),fontsize=15)
        plt.tight_layout()
        plt.savefig(f"DUMMY.jpg",dpi=100)
        plt.clf()


    movieNAME = f"{DATA_DIR}/Trajectory.gif"
    NSKIP     = NSTEPS // NFRAMES
    with imageio.get_writer(movieNAME, mode='I', fps=20) as writer: # Get a writer object
        for frame in range( 0, NSTEPS, NSKIP ):
            print ("Compiling Frame: %1.0f of %1.0f" % ( (frame+1)//NSKIP, NSTEPS//NSKIP) )
            make_frame( frame )
            image = imageio.imread( "DUMMY.jpg" ) # Read JPEG file
            writer.append_data(image) # Write JPEG file (to memory at first; then printed at end)
    sp.call("rm DUMMY.jpg", shell=True)
    gifOPT(movieNAME) # This will compress the GIF movie by at least a factor of two/three. With this: ~750 frames --> 80 MB

def save_data():

    # Create XYZ File
    NSKIP  = NSTEPS // NXYZ
    FILE01 = open(f"{DATA_DIR}/Trajectory.xyz","w")
    for step in range( 0, NSTEPS, NSKIP ):
        FILE01.write(f"{NPARTICLES}\n")
        FILE01.write("MD Step: %1.0f   Time: %1.4f\n" %(step, step*dt))
        for p in range( NPARTICLES ):
            FILE01.write( "X %1.4f %1.4f %1.4f\n" % (COORDS[p,step,0],COORDS[p,step,1],COORDS[p,step,2]) )
    FILE01.close()


def main():
    T1 = time()
    get_Globals()
    get_Initial_COORDS_VELOCS()
    propagate_VV()
    save_data()
    plot()
    make_movie()
    print("\tTotal CPU Time: %1.3f seconds" % (time() - T1) )


if ( __name__ == "__main__" ):
    main()