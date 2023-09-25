import numpy as np
import subprocess as sp

from matplotlib import pyplot as plt

def getGlobals():
    global TIME_GRID, NSTEPS, dT
    global h_t, h_0
    global v_t, v_0
    global GRAV_CONST

    h_0  = 10.0 # Initial Height of Ball
    v_0  = 0.0  # Initial Velocity of Ball

    GRAV_CONST = -1 * 9.8 # g, Gravitational Acceleration of Earth m/s^2

    TMIN   = 0.0
    TMAX   = 20.0
    dT     = 0.01 # Convergence parameter

    TIME_GRID = np.arange( TMIN, TMAX, dT )
    NSTEPS = len(TIME_GRID)
    h_t = np.zeros( NSTEPS ) # Time-dependent Height of Ball
    v_t = np.zeros( NSTEPS ) # Time-dependent Velocity of Ball


    global AIR_FRICTION, FRICTION_CONSTANT
    AIR_FRICTION      = True
    FRICTION_CONSTANT = 0.0 # 0.05 # m/s^2 -- Units of Acceleration

    global E_KINETIC, E_POTENTIAL, E_TOTAL
    E_KINETIC   = np.zeros( NSTEPS )
    E_POTENTIAL = np.zeros( NSTEPS )
    E_TOTAL     = np.zeros( NSTEPS )

    global DATA_DIR
    DATA_DIR = "5.2.1_Bouncing_Ball"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

def Euler_Propagate():

    h_t[0] = h_0
    v_t[0] = v_0

    E_KINETIC[0], E_POTENTIAL[0], E_TOTAL[0] = get_total_energy(h_t[0],v_t[0])


    for step in range( NSTEPS-1 ):

        h_t[step+1] = h_t[step] + dT * v_t[step]
        v_t[step+1] = v_t[step] + dT * GRAV_CONST 

        if ( AIR_FRICTION == True ): # Lose energy due to air friction
            v_t[step+1] = v_t[step+1] - np.sign(v_t[step+1]) * dT * FRICTION_CONSTANT * v_t[step+1]**2

        if ( h_t[step+1] < 0.0 ): # Can't go through the floor...
            #h_t[step+1] *= -1 # Put back to above ground
            v_t[step+1] *= -1 # Switch direction of motion

        E_KINETIC[step+1], E_POTENTIAL[step+1], E_TOTAL[step+1] = get_total_energy(h_t[step+1],v_t[step+1])


def get_total_energy( h, v ):

    # Add up Hamiltonian pieces
    E_KINETIC   = 0.500 * v**2
    E_POTENTIAL = -1 * GRAV_CONST * h
    E_TOTAL     = E_KINETIC + E_POTENTIAL
    return E_KINETIC, E_POTENTIAL, E_TOTAL



def plot():

    plt.plot( TIME_GRID, h_t )
    plt.xlim(TIME_GRID[0],TIME_GRID[-1])
    plt.ylim(0,10)
    plt.xlabel("Time, s", fontsize=15)
    plt.ylabel("Height, m", fontsize=15)
    plt.title("Bouncing Ball", fontsize=15)
    plt.savefig(f"{DATA_DIR}/h_t",dpi=300)
    plt.clf()

    plt.plot( TIME_GRID, v_t, label="Euler" )
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Velocity (m/s)", fontsize=15)
    plt.title("Bouncing Ball", fontsize=15)
    plt.legend()
    plt.xlim(TIME_GRID[0],TIME_GRID[-1])
    plt.savefig(f"{DATA_DIR}/v_t.jpg",dpi=300)
    plt.clf()

    plt.plot( TIME_GRID, E_TOTAL, c="black", label="Total" )
    plt.plot( TIME_GRID, E_KINETIC, c="red", label="Kinetic" )
    plt.plot( TIME_GRID, E_POTENTIAL, c="blue", label="Potential" )
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Energy", fontsize=15)
    plt.legend()
    plt.xlim(TIME_GRID[0],TIME_GRID[-1])
    plt.savefig(f"{DATA_DIR}/E_t.jpg",dpi=300)
    plt.clf()

def main():
    getGlobals()
    Euler_Propagate()
    plot()

    print("\n\tA. Do we bounce ?")
    print("\tB. Do we gain or lose energy ?")
    print("\tC. What if we lost energy when we bouce ? Diss. ~ v(t)")

if ( __name__ == "__main__" ):
    main()