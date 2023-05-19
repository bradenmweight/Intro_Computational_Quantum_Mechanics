import numpy as np

def do_part_a():
    print ( "\n\tStarting part (a):" )
    XGRID = np.linspace( 0, 1, 1000 ) # X coordinates
    dx    = XGRID[1] - XGRID[0]
    PSI   = np.sin( np.pi * XGRID )  # Non-normalized wavefunction
    print ( "\t\tIs PSI normalized (before) ?  %1.6f (Analytic = %1.6f)" % ( np.sum( PSI * PSI ) * dx, 0.500000 )  )
    N2_INV  = np.sum( PSI * PSI ) * dx # Perform integration/inner product --> This is 1/N^2 from homework problem
    N_INV   = np.sqrt( N2_INV )
    N       = 1/N_INV # This is the "N" from the homework problem
    PSI     = N * PSI
    print ( "\t\tIs PSI normalized (after)  ?  %1.6f" % ( np.sum( PSI * PSI ) * dx )  )

def do_part_b():
    print ( "\n\tStarting part (b):" )
    print ( "\t\tFor yoursevles to solve. ")
    print ( "\t\tAnalytic integration: %1.6f" % ( np.sqrt(np.pi) ) )


def do_part_c():
    print ( "\n\tStarting part (c):" )
    print ( "\t\tFor yoursevles to solve. ")
    print ( "\t\tAnalytic integration: %1.6f" % ( 1000/3 ) )
    
def main():
    """
    Notes:
    --> Note that I only used rectangular (Riemann) integration 
            with a large number of points.
    --> You can see that we can match the analytic result 
            up to 6 digits without much trouble. Good enough for us.
    --> I left parts (b) and (c) for you to try. 
            It amounts to copy/paste part (a) and 
            changing the bounds/function.
    """
    do_part_a()
    do_part_b()
    do_part_c()

if ( __name__ == "__main__" ):
    main()

