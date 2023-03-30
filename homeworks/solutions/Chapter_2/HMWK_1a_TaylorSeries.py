import numpy as np
from math import factorial
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess as sp

# Make a data folder
DATA_DIR = "PLOTS_HMWK_1"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def exp_x_taylor_series(N,x): # 0, 1, 2, 3, ...
    """
    Returns Nth-order approximation of f(x) = e^x at x.
    N = 1: exp(x) ~ 1 + x
    N = 2: exp(x) ~ 1 + x + x^2 / 2
    N = 3: exp(x) ~ 1 + x + x^2 / 2 + x^3 / 3!
    """    
    result = 0
    for n in range( N ):
        result += x**n / factorial(n)
    return result

def plot_results(RESULTS, EXACT, x):

    RESULTS = np.array(RESULTS)
    X,Y = RESULTS[:,0], RESULTS[:,1]
    if ( x < 0 ):
        #plt.semilogy(X,np.abs(Y-EXACT),"--o",label=f"x = {round(x,4)}")
        plt.loglog(X,np.abs(Y-EXACT),"--o",label=f"x = {round(x,4)}")
    else:
        #plt.semilogy(X,np.abs(Y-EXACT),"-s",label=f"x = {round(x,4)}")
        plt.loglog(X,np.abs(Y-EXACT),"-s",label=f"x = {round(x,4)}")
    plt.xlabel("Taylor Series Order, N",fontsize=15)
    plt.ylabel("|X$_N$ - X$_{EXACT}$|",fontsize=15)
    plt.title("Taylor Series Convergence: f(x) = e$^x$",fontsize=15)
    plt.legend()
    plt.savefig(f"{DATA_DIR}/Taylor_Series.jpg",dpi=300)


def main():
    x_list       = [-20,-10,-1,1,10,20] # Goal values
    
    for ind,x in enumerate(x_list):
        EXACT_RESULT = np.exp(x)
        N_LIST       = np.arange(1,1005,10)
        RESULTS      = np.zeros(( len(N_LIST), 2 )) # Store N and result

        for count,N in enumerate( N_LIST ):
            RESULTS[count,0] = N
            RESULTS[count,1] = exp_x_taylor_series(N,x)

        plot_results(RESULTS,EXACT_RESULT,x)



if ( __name__ == "__main__" ):
    main()