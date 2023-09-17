import numpy as np
import subprocess as sp

from matplotlib import pyplot as plt

def getGlobals():
    global TIME_GRID, NSTEPS, dT
    global h_0, v_0
    global h_t_EU, h_t_VV
    global v_t_EU, v_t_VV
    global GRAV_CONST

    h_0  = 10.0 # Initial Height of Ball
    v_0  = 0.0  # Initial Velocity of Ball

    GRAV_CONST = -1 * 9.8 # g, Gravitational Acceleration of Earth m/s^2

    TMIN   = 0.0
    TMAX   = 20.0
    dT     = 0.001 # Convergence parameter

    TIME_GRID = np.arange( TMIN, TMAX, dT )
    NSTEPS = len(TIME_GRID)
    h_t_EU = np.zeros( NSTEPS ) # Time-dependent Height of Ball
    h_t_VV = np.zeros( NSTEPS ) # Time-dependent Height of Ball
    v_t_EU = np.zeros( NSTEPS ) # Time-dependent Velocity of Ball
    v_t_VV = np.zeros( NSTEPS ) # Time-dependent Velocity of Ball

    global BOUNCE
    BOUNCE = True # If not bounce, we stop the ball

    global AIR_FRICTION, FRICTION_CONSTANT
    AIR_FRICTION      = True
    FRICTION_CONSTANT = 0.1 # 1/s -- Units of Acceleration

    global E_KINETIC_EU, E_POTENTIAL_EU, E_TOTAL_EU
    global E_KINETIC_VV, E_POTENTIAL_VV, E_TOTAL_VV
    E_KINETIC_EU   = np.zeros( NSTEPS )
    E_POTENTIAL_EU = np.zeros( NSTEPS )
    E_TOTAL_EU     = np.zeros( NSTEPS )
    E_KINETIC_VV   = np.zeros( NSTEPS )
    E_POTENTIAL_VV = np.zeros( NSTEPS )
    E_TOTAL_VV     = np.zeros( NSTEPS )

    global STOP
    STOP = NSTEPS-1


    global DATA_DIR
    DATA_DIR = "5.2.2_Bouncing_Ball_Velocity_Verlet"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

def Euler_Propagate():

    h_t_EU[0] = h_0
    v_t_EU[0] = v_0

    E_KINETIC_EU[0], E_POTENTIAL_EU[0], E_TOTAL_EU[0] = get_total_energy(h_t_EU[0],v_t_EU[0])

    for step in range( NSTEPS-1 ):

        h_t_EU[step+1] = h_t_EU[step] + dT * v_t_EU[step]
        v_t_EU[step+1] = v_t_EU[step] + dT * GRAV_CONST 

        if ( AIR_FRICTION == True ): # Lose energy due to air friction
            v_t_EU[step+1] = v_t_EU[step+1] - np.sign(v_t_EU[step+1]) * dT * FRICTION_CONSTANT * v_t_EU[step+1]**2

        E_KINETIC_EU[step+1], E_POTENTIAL_EU[step+1], E_TOTAL_EU[step+1] = get_total_energy(h_t_EU[step+1],v_t_EU[step+1])

        if ( h_t_EU[step+1] < 0.0 ): # Can't go through the floor...
            if ( BOUNCE == True ):
                v_t_EU[step+1] *= -1 
            else:
                STOP = step+1
                break # Stop the simulation



def Velocity_Verlet_Propagate():

    h_t_VV[0] = h_0
    v_t_VV[0] = v_0

    E_KINETIC_VV[0], E_POTENTIAL_VV[0], E_TOTAL_VV[0] = get_total_energy(h_t_VV[0],v_t_VV[0])


    for step in range( NSTEPS-1 ):

        # Half-step Position
        h_t_VV[step+1] = h_t_VV[step] + 0.5000 * dT * v_t_VV[step]
        # Full-step Velocity
        v_t_VV[step+1] = v_t_VV[step] + dT * GRAV_CONST
        # Half-step Position
        h_t_VV[step+1] = h_t_VV[step+1] + 0.5000 * dT * v_t_VV[step+1] 
            # Uses updated velocity for second half-step in position
            # O(dt) --> O(dt^2) for this simple trick !!!

        # Approximate addition of air friction. We could add to VV scheme if we wanted to be better.
        if ( AIR_FRICTION == True ): # Lose energy due to air friction
            v_t_VV[step+1] -= np.sign(v_t_VV[step+1]) * dT * FRICTION_CONSTANT * v_t_VV[step+1]**2

        E_KINETIC_VV[step+1], E_POTENTIAL_VV[step+1], E_TOTAL_VV[step+1] = get_total_energy(h_t_VV[step+1],v_t_VV[step+1])

        if ( h_t_VV[step+1] < 0.0 ): # Can't go through the floor...
            if ( BOUNCE == True ):
                v_t_VV[step+1] *= -1 
            else:
                STOP = step+1
                break # Stop the simulation


def get_total_energy( h, v ):

    # Add up Hamiltonian pieces
    E_KINETIC   = 0.500 * v**2
    E_POTENTIAL = -1 * GRAV_CONST * h
    E_TOTAL     = E_KINETIC + E_POTENTIAL
    return E_KINETIC, E_POTENTIAL, E_TOTAL

def plot():

    plt.plot( TIME_GRID[:STOP], h_t_EU[:STOP], label="Euler" )
    plt.plot( TIME_GRID[:STOP], h_t_VV[:STOP], label="Velocity-Verlet" )
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Height (m)", fontsize=15)
    plt.title("Bouncing Ball ???", fontsize=15)
    plt.legend()
    plt.savefig(f"{DATA_DIR}/h_t.jpg",dpi=300)
    plt.clf()

    plt.plot( TIME_GRID[:STOP], v_t_EU[:STOP], label="Euler" )
    plt.plot( TIME_GRID[:STOP], v_t_VV[:STOP], label="Velocity-Verlet" )
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Velocity (m/s)", fontsize=15)
    plt.title("Bouncing Ball ???", fontsize=15)
    plt.legend()
    plt.savefig(f"{DATA_DIR}/v_t.jpg",dpi=300)
    plt.clf()

    plt.plot( TIME_GRID[:STOP], E_TOTAL_EU[:STOP], "--", c="black", label="Total (Euler)" )
    plt.plot( TIME_GRID[:STOP], E_TOTAL_VV[:STOP], "-", c="black", label="Total (Vel-Ver)" )
    plt.plot( TIME_GRID, E_KINETIC_EU, "--", c="red", label="Kinetic (Euler)" )
    plt.plot( TIME_GRID, E_KINETIC_VV, "-", c="red", label="Kinetic (Vel-Ver)" )
    plt.plot( TIME_GRID, E_POTENTIAL_EU, "--", c="blue", label="Potential (Euler)" )
    plt.plot( TIME_GRID, E_POTENTIAL_VV, "-", c="blue", label="Potential (Vel-Ver)" )
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Energy", fontsize=15)
    plt.legend()
    plt.savefig(f"{DATA_DIR}/E_t.jpg",dpi=300)
    plt.clf()


def main():
    getGlobals()
    Euler_Propagate()
    Velocity_Verlet_Propagate()
    plot()

    print("\n\tA. Do we bounce ?")
    print("\tB. Do we gain or lose energy ?")
    print("\tC. What if we lost energy when we bouce ? Diss. ~ v(t)")

if ( __name__ == "__main__" ):
    main()