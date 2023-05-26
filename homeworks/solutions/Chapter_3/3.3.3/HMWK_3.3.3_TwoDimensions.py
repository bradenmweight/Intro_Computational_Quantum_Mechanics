import numpy as np
from scipy.special import hermite
from scipy.integrate import trapezoid

def get_globals():

    Nx = 10**2 # Do not go higher than 10**4
    Ny = 10**2 # Do not go higher than 10**4

    check_memory( Nx, Ny ) # Multi-dimensional function eat a lot of memory. Be careful.

    global X, Y
    X = np.linspace( -0.5, 1.0, Nx )
    Y = np.linspace( -1.0, 1.0, Ny )

    global dX, dY
    dX = X[1] - X[0]
    dY = Y[1] - Y[0]

def check_memory( Nx, Ny ):

    UNIT_SIZE    = 8 # Size of float
    TOTAL_MEMORY = UNIT_SIZE * Nx * Ny # Size of unit multiplied by the number of elements

    MB_SIZE = TOTAL_MEMORY * 10 ** -6
    GB_SIZE = TOTAL_MEMORY * 10 ** -9

    print( "\nTotal Memory: %1.2f MB, %1.2f GB " % ( MB_SIZE, GB_SIZE ) )
    if ( GB_SIZE  > 1 ):
        print("\n\tWarning !!! Need too much memory (MEM > 1GB). Killing job for safety.\n")

def my_function():

    """
    # Slow Version (Non-pythonic Way)
    f_x_y = np.zeros(( len(X), len(Y) ))
    for xi,x in enumerate(X):
        for yi,y in enumerate(Y):
            f_x_y[xi,yi] = np.sin( x**3 * y**2 ) ** 2
    return f_x_y
    """

    # Faster Version (Pythonic Way)
    XG,YG = np.meshgrid(X,Y)
    f_x_y = np.sin( XG**3 * YG**2 ) ** 2
    return f_x_y



def integrate_2D( f_x_y ):
    return np.sum( f_x_y[:,:] ) * dX * dY

    # An alternative way to do a sum
    # This is a more general way to do linear algebra in python
    #return np.einsum( "jk->", f_x_y[:,:]  ) * dX * dY

def main():
    get_globals()
    f_x_y = my_function()
    I = integrate_2D( f_x_y )
    print ( "I = %1.6f (Exact: %1.6f)" % (I, 0.0522359) )

if ( __name__ == "__main__" ):
    main()

