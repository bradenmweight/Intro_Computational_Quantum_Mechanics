import numpy as np
from scipy.special import hermite

def S_2x2():
    # Define grid and spacing
    X     = np.linspace( -20,20,10**4 )
    dX    = X[1] - X[0]

    # Get the Hermite polynomials for ground (0) and excited (1) state
    H_0   = hermite(0) # This is a function. Need to call it yet.
    H_1   = hermite(1) # This is a function. Need to call it yet.

    # Get wavefunctions for ground (0) and excited (1) states
    PSI_0 = np.exp( -X**2 / 2 ) * H_0(X)
    PSI_1 = np.exp( -X**2 / 2 ) * H_1(X)

    # Construct overlap matrix for ground (0) and excited (1) states
    S = np.zeros((2,2))
    S[0,0] = np.sum( PSI_0 ** 2 ) * dX
    S[1,1] = np.sum( PSI_1 ** 2 ) * dX
    S[0,1] = np.sum( PSI_0 * PSI_1 ) * dX
    S[1,0] = S[0,1] # Overlap matrix must be symmetric
    print( "Overlap Matrix (Before Normalization):\n", np.round(S,4) )

    # Normalize the wavefunctions
    PSI_0 /= np.sqrt( S[0,0] )
    PSI_1 /= np.sqrt( S[1,1] )

    # Reconstruct overlap matrix
    S = np.zeros((2,2))
    S[0,0] = np.sum( PSI_0 ** 2 ) * dX
    S[1,1] = np.sum( PSI_1 ** 2 ) * dX
    S[0,1] = np.sum( PSI_0 * PSI_1 ) * dX
    S[1,0] = S[0,1] # Overlap matrix must be symmetric
    print( "Overlap Matrix (After Normalization):\n", np.round(S,4) )


def S_NxN( N ):
    # Define grid and spacing
    X     = np.linspace( -20,20,10**4 )
    dX    = X[1] - X[0]

    PSI_N = np.zeros(( N, len(X) ))
    S     = np.zeros(( N, N ))
    for n in range( N ):
        for m in range( N ):
            # Get the Hermite polynomials for ground (0) and excited (1) state
            H_n   = hermite(n) # This is a function. Need to call it yet.
            H_m   = hermite(m) # This is a function. Need to call it yet.

            # Get wavefunctions for ground (0) and excited (1) states
            PSI_n = np.exp( -X**2 / 2 ) * H_n(X)
            PSI_m = np.exp( -X**2 / 2 ) * H_m(X)

            # Normalize both wavefunctions
            PSI_n /= np.sqrt( np.sum( PSI_n **2 ) * dX )
            PSI_m /= np.sqrt( np.sum( PSI_m **2 ) * dX )

            # Construct overlap matrix for ground (0) and excited (1) states
            S[n,m] = np.sum( PSI_n * PSI_m ) * dX
            print( n,m,np.round(S[n,m],4) )


    print( np.round(S,4) )

def main():
    S_2x2()
    S_NxN( 5 )

if ( __name__ == "__main__" ):
    main()

