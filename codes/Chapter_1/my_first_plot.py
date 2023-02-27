# my_first_plot.py

# Python modules
import random

# External modules
import numpy as np 
from matplotlib import pyplot as plt 

# User-made modules
# We don't yet have any of these.

def get_globals():
    global X_GRID
    
    # Create a numpy array starting from 1 to 1000 in increments of 2
    X_GRID = np.arange( 1,1001,2 ) 


def get_Y_quadradtic():
    """
    The purpose of this function is to create the Y-values for a user-defined function
    INPUT:  None
    OUTPUT: Y [1D nd.array]
    """

    # Let's do a quadratic function
    Y_X = X_GRID ** 2 / 10**3 # y(x) = x^2 / 10^4

    # This is a return statement. 
    return Y_X

def plot_quadratic_1D_LINE(Y_X):
    """
    The purpose of this function is to generate a 1D line plot.
      Generates image file "Y_X_LINE.jpg"
    
    INPUT:  Y_X [1D nd.array]
    OUTPUT: None
    """

    plt.plot( X_GRID, Y_X, "-", c='black', linewidth=3, label="My 1D curve:\n$Y(X) = \\frac{X^2}{10^3} $" )
    
    ##### OPTIONAL PLOTTING COMMANDS #####
    plt.legend()
    plt.xlim( X_GRID[0], X_GRID[-1] ) # Define x-axis as limits of your chosen X_GRID
    plt.ylim( 0 ) # Define one limit of the y-axis
    plt.xlabel("This is my X-Axis",fontsize=15)
    plt.ylabel("This is my Y-Axis",fontsize=15)
    plt.title("This is my title",fontsize=15)
    ######################################

    plt.savefig("Y_X_LINE.jpg",dpi=600)
    plt.clf() # This clears the internal plot in case we want to make another.

def get_Y_random():
    """
    The purpose of this function is to create the Y-values for a user-defined function
    INPUT:  None
    OUTPUT: Y [1D nd.array]
    """

    # How many Y-values do we need to get ?
    NGRID = len(X_GRID)

    #### Let's do a random function
    # These are uniform random numbers on the interval [0,1)
    RAND = random.random() # This is a single random number
    RAND_LIST = [ 2*random.random() - 1 for j in range(NGRID) ] # This is a list of random numbers
    # "for j in range(NGRID)" above just gives us the same number of randoms as the X_GRID variable
    # "2*random.random() - 1" converts from [0,1) to [-1,1)

    # Let's also do a set of random gaussian numbers with mean of 0 and width of 1.
    RAND_GAUSS_LIST = [ random.gauss(0,1) for j in range(NGRID) ]

    return RAND_LIST, RAND_GAUSS_LIST


def plot_random_1D_LINE_SCATTER_two_FUNCTIONS(Y1, Y2):
    """
    The purpose of this function is to generate a 1D line plot.
      Generates image file "Y_X_LINE.jpg"
    
    INPUT:  Y1 [1D nd.array]
    INPUT:  Y2 [1D nd.array]
    OUTPUT: None
    """

    plt.scatter( X_GRID, Y1, c="blue", label="Uniform Random" )
    plt.plot( X_GRID, Y2, "o", c="red", label="Guassian Random", alpha=0.25 )
    
    # Plot horzontal lines indicating certain quantities
    plt.plot( X_GRID, 1.0 *np.ones(len(X_GRID)), "--", c="black", lw=2, label="$\pm 1$" )
    plt.plot( X_GRID, -1.0*np.ones(len(X_GRID)), "--", c="black", lw=2 )



    ##### OPTIONAL PLOTTING COMMANDS #####
    plt.legend()
    plt.xlim( 0, 1000 ) # Define x-axis as limits of your chosen X_GRID
    plt.ylim( -5,5 ) # Define limits of the y-axis
    plt.xlabel("Random Coordinate",fontsize=15)
    plt.ylabel("Random Value",fontsize=15)
    plt.title("Uniform vs. Gaussian Random Numbers",fontsize=15)
    ######################################

    plt.savefig("Y_X_RANDOM_SCATTER.jpg",dpi=600)
    plt.clf() # This clears the internal plot in case we want to make another.

def main():
    
    get_globals()
    
    # Let's plot a quadratic function
    Y_X = get_Y_quadradtic()
    plot_quadratic_1D_LINE(Y_X)

    # Let's plot a random function
    RAND_LIST, RAND_GAUSS_LIST = get_Y_random()
    plot_random_1D_LINE_SCATTER_two_FUNCTIONS(RAND_LIST, RAND_GAUSS_LIST)

if ( __name__ == "__main__" ):
    main()