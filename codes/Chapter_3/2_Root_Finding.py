import numpy as np
from scipy.optimize import newton

def get_globals():
    global TOL, MAX_STEP, XGRID
    TOL      = 10 ** -20                   # Stop search when {TOL} tolerance is reached
    MAX_STEP = 10 ** 4                     # Stop after {MAX_STEP} steps
    XGRID    = np.linspace(0,2*np.pi,1000) # Function grid
    
    # The exact root of the function for comparison
    global EXACT_ROOT
    EXACT_ROOT = np.pi

    # Bisection method requires initial bounds
    global a0,b0
    a0 = 0 # Left bound
    b0 = 6 # Right bound

    # Newton-Raphson (Secant) method requires 
    #    initial position and guess for next position
    global x0, x1
    x0 = 0
    x1 = 6

    # Scipy needs an initial guess for the root
    global root_guess
    root_guess = 1


def get_function(x):
    """
    Evaluate function at x.
    INPUT:    x (float)
    OUTPUT: f_x (float)
    """
    f_x = np.sqrt( x + 1 ) * np.cos( x / 2 ) ** 3
    return f_x

def do_Bisection_method():
    """
    Perform bisection method for root-finding
    INPUT:  None
    OUTPUT: Approximate root of function (float)
    """
    # Since we need to modify x0 and x1,
    #   we need this line to appear here
    
    fa0 = get_function(a0)
    fb0 = get_function(b0)

    if ( fa0*fb0 > 0 ):
        print("\n\tWARNING:")
        print(f"\tThere is no root between (a0,b0) = ({a0},{b0})")
        print("\tChoose new initial bounds.\n")
        #return

    an = a0
    bn = b0

    for step in range( MAX_STEP ):
        # Store function and interval midpoint values
        fan      = get_function(an)
        fbn      = get_function(bn)
        cn       = (an + bn) / 2
        fcn      = get_function(cn)

        # Check convergence based on interval
        if ( abs(an) <= 1e-30 ):
            INTERVAL = abs(bn-an)
        else:
            INTERVAL = abs(bn-an) / abs(an)
        if ( INTERVAL <= TOL ):
            print(f"\n\tBisection Method:")
            print(f"\tRoot converged after {step+1} iterations:")
            print(f"\tRoot: {cn}\n")
            return cn # Exit FOR-LOOP and return to main()

        # Get next value
        if ( fan*fcn < 0 ):
            bn = cn
        elif ( fbn*fcn < 0 ):
            an = cn
        else:
            print("No root region found. Choosing smaller region.")
            L = bn - an
            an += L/8
            bn -= L/8
            fan = get_function(an)
            fbn = get_function(bn)
            print(an,bn,fan*fbn)

    print(f"\n\tFAILURE: Root not converged after {MAX_STEP} iterations.")
    print(f"\tFinal root: {cn}")
    return cn

def do_Secant_method():
    """
    Perform Newton-Raphson (Secant) method for root-finding
    INPUT:  None
    OUTPUT: Approximate root of function (float)
    """
    # Since we need to modify x0 and x1,
    #   we need this line to appear here
    global x0,x1 
    
    for step in range( MAX_STEP ):
        # Store function values
        fx0 = get_function(x0)
        fx1 = get_function(x1)

        # Get next value
        x2 = x1 - fx1 * (x1-x0) / (fx1-fx0)

        # Check convergence based on interval
        INTERVAL = abs(x2-x1) / abs(x1)
        if ( INTERVAL <= TOL ):
            print(f"\n\tNewton-Raphson (Secant) Method:")
            print(f"\tRoot converged after {step+1} iterations:")
            print(f"\tRoot: {x2}\n")
            return x2 # Exit FOR-LOOP and return to main()

        # Set up for next iteration
        x0 = x1
        x1 = x2

    print(f"\n\tFAILURE: Root not converged after {MAX_STEP} iterations.")
    print(f"\tFinal root: {x2}")
    return x2

def do_SCIPY_method():
    results = newton( get_function, root_guess, maxiter=MAX_STEP, tol=TOL, rtol=TOL, full_output=True )
    print(f"\n\tSCIPY Newton-Raphson (Secant) Method:")
    print(f"\tRoot converged after {results[1].iterations} iterations:")
    print(f"\tRoot: {results[0]}\n")
    return results[0]

def get_errors(root_BISECT,root_SECANT,root_SCIPY):

    print(f"\n\tErrors in each method at TOLERANCE = {TOL}:")
    print(f"\tExact root:           {EXACT_ROOT}")
    print(f"\tBisection:            {abs(EXACT_ROOT-root_BISECT)/(EXACT_ROOT)*100} %")
    print(f"\tNewton-Raphson:       {abs(EXACT_ROOT-root_SECANT)/(EXACT_ROOT)*100} %")
    print(f"\tSCIPY Newton-Raphson: {abs(EXACT_ROOT-root_SCIPY)/(EXACT_ROOT)*100} %\n")

    print("\tBisection (%1.2e %s):            %1.30f" % (abs(EXACT_ROOT-root_BISECT)/(EXACT_ROOT)*100,"%",abs(EXACT_ROOT-root_BISECT)/(EXACT_ROOT)))
    print("\tNewton-Raphson (%1.2e %s):       %1.30f" % (abs(EXACT_ROOT-root_SECANT)/(EXACT_ROOT)*100,"%",abs(EXACT_ROOT-root_SECANT)/(EXACT_ROOT)))
    print("\tSCIPY Newton-Raphson (%1.2e %s): %1.30f" % (abs(EXACT_ROOT-root_SCIPY)/(EXACT_ROOT)*100,"%",abs(EXACT_ROOT-root_SCIPY)/(EXACT_ROOT)))



def main():
    get_globals()
    root_BISECT = do_Bisection_method()
    root_SECANT = do_Secant_method()
    root_SCIPY  = do_SCIPY_method()
    get_errors(root_BISECT,root_SECANT,root_SCIPY)

if ( __name__ == "__main__" ):
    main()