import numpy as np
from matplotlib import pyplot as plt

def get_Params():
    global xGRID, Nx, dx
    xMIN  = -10
    xMAX  =  10
    Nx    = 1001
    xGRID = np.linspace(xMIN, xMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    ### Define current system ###
    global Vx, WFN_0, E_0
    CASE = "QHO" # "ISW" or "QHO"


    if ( CASE == "QHO" ):
        # QHO
        w     = 0.15000
        Vx    = 0.5000 * w**2 * xGRID**2 
        WFN_0 = np.exp( -w * xGRID**2 / 2 )
        WFN_0 = WFN_0 / np.linalg.norm( WFN_0 )
        E_0   = 0.5000 * w

    elif ( CASE == "ISW" ):

        # Infinite square well
        L     = 4 # Width of the well
        Vx    = np.zeros( (Nx) )
        WFN_0 = np.cos( xGRID * np.pi / 2 )
        for xi,x in enumerate(xGRID):
            if ( x < -L/2 or x > L/2 ):
                Vx[xi] = 1000
                WFN_0[xi] = 0.0
            else:
                Vx[xi] = 0.0
                WFN_0[xi] = np.cos( x * np.pi / L )
        WFN_0 = WFN_0 / np.linalg.norm( WFN_0 )
        E_0   = np.pi**2 / 2 / L**2
    #############################

def get_Tx( N, d ):
    """
    Construct the kinetic energy operator.
    """
    T  = np.zeros((N, N))
    T -= np.diag( np.ones(N-1), k=-1 )
    T += np.diag( 2*np.ones(N), k=0 )
    T -= np.diag( np.ones(N-1), k=1 )
    T = T / 2 / d**2
    return T

def get_Basis( NBASIS ):
    """
    Choose quantum harmonic oscillator basis functions.
    These are usually analytic, but we can also use numerical.
    Could be Gaussians, Slater-type orbitals, etc.
    """
    w = 5.0000
    V = 0.5000 * w**2 * np.diag( xGRID**2 )
    T = get_Tx( Nx, dx )
    _, U = np.linalg.eigh( T + V )
    return U[:,:NBASIS]

def get_Potential_Matrix_Elements( Ubasis ):
    """
    Calculate the matrix elements of the potential operator.
    """
    V  = np.einsum("xj,x,xk->jk", Ubasis, Vx, Ubasis ) # jk matrix elements of \hat{V}
    return V

def get_Kinetic_Matrix_Elements( Ubasis ):
    """
    Calculate the matrix elements of the kinetic operator.
    """
    T = np.einsum("xj,xy,yk->jk", Ubasis, get_Tx( Nx, dx ), Ubasis ) # jk matrix elements of \hat{T}
    return T

def plot_Single( E, U, Ubasis ):
    Ux    = Ubasis @ U[:,0] # Rotate to position basis from QHO basis
    PHASE = np.sign( np.sum( Ux * WFN_0[:] ) ) # Check phase of exact vs. numerical
    for i in range( len(Ubasis[0,:]) ):
        #plt.plot(xGRID, Ubasis[:,i], lw=1, alpha=0.25)
        plt.plot(xGRID, np.abs(U[i,0]) * Ubasis[:,i], lw=1, alpha=0.5)
    plt.plot(xGRID, WFN_0, c="black", lw=8, alpha=0.25, label="Exact GS")
    plt.plot(xGRID, Ux * PHASE, c="red", lw=2, label="Numerical GS")
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Energy / Wavefunction", fontsize=15)
    plt.title("$E_0^\\mathrm{Num.}$ = %1.3f   $E_0^\\mathrm{Exact}$ = %1.3f    NBASIS = %d" % (E[0], E_0, len(Ubasis[0,:])), fontsize=15)
    plt.legend()
    #plt.xlim(-4,4)
    plt.tight_layout()
    plt.savefig("Ground_State_WFN.jpg", dpi=300)
    plt.clf()

def plot_scan( E_GS, nbasis_list ):
    plt.plot(nbasis_list, nbasis_list*0 + E_0, "--", c="black", lw=3, label="Numerical")
    plt.plot(nbasis_list, E_GS, c="red", lw=3, label="Exact")
    plt.legend()
    plt.xlabel("Number of Basis Functions", fontsize=15)
    plt.ylabel("Ground State Energy", fontsize=15)
    plt.title("Convergence of Ground State Energy", fontsize=15)
    plt.savefig("Ground_State_Convergence.jpg", dpi=300)
    plt.clf()

    plt.loglog(nbasis_list, E_GS - E_0, c="black", lw=3)
    plt.xlabel("Number of Basis Functions", fontsize=15)
    plt.ylabel("Ground State Energy", fontsize=15)
    plt.title("Convergence of Ground State Energy", fontsize=15)
    plt.savefig("Ground_State_Convergence_Error.jpg", dpi=300)
    plt.clf()

def main():

    # First do single calculation with fixed basis size
    nbasis = 50
    print("Working on basis size: %d" % nbasis)
    get_Params()
    Ubasis = get_Basis( nbasis )
    V      = get_Potential_Matrix_Elements( Ubasis )
    T      = get_Kinetic_Matrix_Elements( Ubasis )
    H      = T + V
    E, U   = np.linalg.eigh( T + V )
    print("Basis size: %d   Energy: %1.6f    Exact: %1.6f" % (nbasis, E[0], E_0))
    plot_Single( E, U, Ubasis )



    # Do a scan over the number of basis functions until convergence
    nbasis_list = np.arange( 1,500,1 )
    E_GS        = np.zeros( len(nbasis_list) )
    Ubasis      = get_Basis( nbasis_list[-1] )
    print("Calculating V matrix elements...fast")
    V           = get_Potential_Matrix_Elements( Ubasis )
    print("Calculating T matrix elements...slow")
    T           = get_Kinetic_Matrix_Elements( Ubasis )
    for bi,b in enumerate(nbasis_list):
        T_tmp  = T[:b,:b]
        V_tmp  = V[:b,:b]
        H      = T_tmp + V_tmp
        E, U   = np.linalg.eigh( H )
        print("Basis size: %d   Energy: %1.6f    Exact: %1.6f" % (b, E[0], E_0))
        E_GS[bi] = E[0]
    plot_scan( E_GS, nbasis_list )

if ( __name__ == "__main__" ):
    main()