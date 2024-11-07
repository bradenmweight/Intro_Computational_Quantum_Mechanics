import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import subprocess as sp

DATA_DIR = "8.2_TDSE_EnergyBasis_Position"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Globals():
    global Nx, dx, xGRID
    XMIN  = -10.0
    XMAX  = 10.0
    Nx    = 2000
    xGRID = np.linspace(XMIN, XMAX, Nx)
    dx    = xGRID[1] - xGRID[0]

    global dt, tGRID
    tMAX  = 8.0
    dt    = 0.5
    tGRID = np.arange(0, tMAX+dt, dt)


def propagate( E, psi_0, name="" ):
    ### Propagate the wavefunction in time
    psi_t = np.zeros( (len(tGRID), len(E)), dtype=np.complex128 )
    E_t   = np.zeros( len(tGRID) )
    for ti,t in enumerate( tGRID ):
        # Rather than diag(E), we can just do element-wise vector multiplication
        # This is faster since all off-diagonal elements of the diag(E) are zero
        psi_t[ti,:] = np.exp( -1j * E[:] * t ) * psi_0[:]
        E_t[ti]     = np.average( E[:] * psi_t[ti,:] ).real
    return psi_t, E_t

def moving_Gaussian( E, U ):

    ### Define the initial wavefunction 
    psi_0      = np.zeros( Nx, dtype=np.complex128 )
    psi_0      = np.exp( -(xGRID-1)**2 / 2 / 0.5**2 )
    psi_0      = psi_0 / np.linalg.norm(psi_0)
    psi_0      = np.einsum("xE,x->E", U, psi_0)
    psi_t, E_t = propagate( E, psi_0 ) # < E_n | psi (t) >
    psi_t      = np.einsum("xE,tE->tx", U[:,:], psi_t[:,:]) # < x | psi (t) >

    cmap = 'viridis'
    normalize = matplotlib.colors.Normalize(vmin=tGRID[0],vmax=tGRID[-1])
    colors = matplotlib.colormaps[cmap](normalize(tGRID))
    fig, ax = plt.subplots()
    plt.plot(np.arange(-2,2,0.001), 0.5 * np.arange(-2,2,0.001)**2, c="black", lw=8, alpha=0.5)
    norm   = np.max( np.abs(psi_t[:,:]) )
    for ti,t in enumerate( tGRID ):
        function = np.abs(psi_t[ti,:])/norm + E_t[ti]
        plt.plot( xGRID, function, "-", c=colors[ti], alpha=0.4, lw=2 )
    for ti,t in enumerate( tGRID ):
        function = np.abs(psi_t[ti,:])/norm + E_t[ti]
        ind_max = np.argmax(function)
        plt.scatter( xGRID[ind_max], function[ind_max], s=40, color=colors[ti], label="MAX[$\\psi (t)$]" * (ti==0) )
    plt.legend()
    mappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)# Make mappable for colorbar
    plt.colorbar(mappable,ax=ax,pad=0.01,label='Time (a.u.)')
    plt.xlim(-2,2)
    plt.xlabel("Position (a.u.)", fontsize=15)
    plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/psi_xt_moving_Gaussian.jpg", dpi=300)
    plt.clf()

def stationary_Gaussian( E, U ):

    ### Define the initial wavefunction 
    psi_0      = U[:,0] # Set as the QHO ground state Gaussian
    psi_0      = np.einsum("xE,x->E", U, psi_0)
    psi_t, E_t = propagate( E, psi_0 ) # < E_n | psi (t) >
    psi_t      = np.einsum("xE,tE->tx", U[:,:], psi_t[:,:]) # < x | psi (t) >

    cmap = 'viridis'
    normalize = matplotlib.colors.Normalize(vmin=tGRID[0],vmax=tGRID[-1])
    colors = matplotlib.colormaps[cmap](normalize(tGRID))
    fig, ax = plt.subplots()
    plt.plot(np.arange(-2,2,0.001), 0.5 * np.arange(-2,2,0.001)**2, c="black", lw=8, alpha=0.5)
    norm   = np.max( np.abs(psi_t[:,:]) )
    for ti,t in enumerate( tGRID ):
        function = np.abs(psi_t[ti,:])/norm + E_t[ti]
        plt.plot( xGRID, function, "-",  c='black', lw=2, label="ABS" * (ti==0) )
        function = np.real(psi_t[ti,:])/norm + E_t[ti]
        plt.plot( xGRID, function, "-",  c=colors[ti], alpha=0.4, lw=2, label="REAL" * (ti==0) )
        #function = np.imag(psi_t[ti,:])/norm + E_t[ti]
        #plt.plot( xGRID, function, "--", c=colors[ti], alpha=0.4, lw=2, label="IMAG" * (ti==0) )
    plt.legend()
    mappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)# Make mappable for colorbar
    plt.colorbar(mappable,ax=ax,pad=0.01,label='Time (a.u.)')
    plt.xlim(-2,2)
    plt.xlabel("Position (a.u.)", fontsize=15)
    plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/psi_xt_stationary_Gaussian_REAL.jpg", dpi=300)
    plt.clf()

    cmap = 'viridis'
    normalize = matplotlib.colors.Normalize(vmin=tGRID[0],vmax=tGRID[-1])
    colors = matplotlib.colormaps[cmap](normalize(tGRID))
    fig, ax = plt.subplots()
    plt.plot(np.arange(-2,2,0.001), 0.5 * np.arange(-2,2,0.001)**2, c="black", lw=8, alpha=0.5)
    norm   = np.max( np.abs(psi_t[:,:]) )
    for ti,t in enumerate( tGRID ):
        function = np.abs(psi_t[ti,:])/norm + E_t[ti]
        plt.plot( xGRID, function, "-",  c='black', lw=2, label="ABS" * (ti==0) )
        function = np.imag(psi_t[ti,:])/norm + E_t[ti]
        plt.plot( xGRID, function, "-", c=colors[ti], alpha=0.4, lw=2, label="IMAG" * (ti==0) )
    plt.legend()
    mappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)# Make mappable for colorbar
    plt.colorbar(mappable,ax=ax,pad=0.01,label='Time (a.u.)')
    plt.xlim(-2,2)
    plt.xlabel("Position (a.u.)", fontsize=15)
    plt.ylabel("Wavefunction, $\\psi(x,t) = \\langle x | \\psi(t) \\rangle$", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/psi_xt_stationary_Gaussian_IMAG.jpg", dpi=300)
    plt.clf()

def main():

    ### Get the energy basis by 
    ###   diagonalizing the Hamiltonian
    get_Globals()
    E, U = get_Energy_Basis()
    print( "QHO Energies:", E[:5] )

    ### Example #1 -- A moving Gaussian
    ### Set parameters to the following
    ### tMAX  = 3.0
    ### dt    = 0.05
    moving_Gaussian( E, U )

    ### Example #2 -- A stationary Gaussian
    ### Set parameters to the following
    ### tMAX  = 8.0
    ### dt    = 0.5
    stationary_Gaussian( E, U )







def get_Energy_Basis():
    def get_V_x():
        return np.diag( 0.5 * xGRID**2 )

    def get_T_DVR():
        T = np.zeros((Nx,Nx))
        for i in range( Nx ):
            T[i,i] = np.pi**2 / 3
            for j in range( i+1, Nx ):
                T[i,j] = (-1)**(i-j) * 2 / (i-j)**2
                T[j,i] = T[i,j]
        return T  / 2 / dx**2
    H = get_T_DVR() + get_V_x()
    E,U = np.linalg.eigh(H)
    return E,U

if ( __name__ == "__main__" ):
    main()

