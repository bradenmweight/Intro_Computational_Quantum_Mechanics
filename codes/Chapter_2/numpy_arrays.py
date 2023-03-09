import numpy as np
import random
from scipy.linalg import expm as SCIPY_MAT_EXP

def learn_make_numpy_array():
    """
    In this function, we will learn about creating arrays.
    """

    print("\n####### BEGIN MAKE AND COMPARE ARRAYS  #######\n")

    # This is a one-dimensional list -- a built-in data type
    my_1D_list        = [1,2,3,4]
    print(f"my_1D_list        = {my_1D_list}")

    # This converts the list to a numpy array
    #   and is usually the easiest way to create
    #   a simple numpy array
    my_1D_numpy_array = np.array(my_1D_list)
    print(f"my_1D_numpy_array = {my_1D_numpy_array}")
    print("\nObservations:")
    print("\tNote the difference in the way they are printed.")
    print("\tNumpy arrays usually do not have commas separating the values.")

    print("\nMaking two-dimensional arrays can be the same as for lists.")
    my_2D_list  = [[1,2],[3,4]]
    my_2D_array = np.array(my_2D_list)
    print(f"my_2D_list = {my_2D_list}")
    print(f"my_2D_array = \n{my_2D_array}")
    print("\nObservations:")
    print("\tNote the difference in the way they are printed.")
    print("\tNumpy arrays attempt to print themselves like matrices when possible \
                \n\t  -- because that's what they are.")

    print("\nMaking higher-dimensional arrays can be the same as for lists \
                but is more complicated to write down.")
    my_3D_list   = [ [ [1,2] ,[3,4]  ],\
                     [ [5,6] ,[7,8]  ],\
                     [ [9,10],[11,12]] ]

    my_3D_array  = np.array(my_3D_list)
    print(f"my_3D_list = {my_3D_list}")
    print(f"my_3D_array = \n{my_3D_array}")
    print("\n\tHow can we examine these better ?")
    print("\tLet's use of for-loop to print them.")
    print("\tLet's iterate over the first index:")
    shape = my_3D_array.shape # Will return (3,2,2)
    print(f"The shape of my array is: {shape}")
    for x in range(shape[0]):
        print(x, "\n", my_3D_array[x])

    print("\n\tLet's iterate over the first and second index:")
    for x in range(shape[0]):
        for y in range(shape[1]):
            print(x,y,my_3D_array[x,y])

    print("\n\tLet's iterate over the first, second, and third index:")
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                print(x,y,z,my_3D_array[x,y,z])

    print("\nNow, can we do the opposite ? Compose the numpy array with a for-loop ?")
    print("We first need an empty array like we needed an empty list: A = np.zeros(shape)")
    A = np.zeros( shape ) # = np.zeros( ( 3,2,2 ) )
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                counter = x*shape[2]*shape[1] + y*shape[1] + z # 0,1,2,3,4,...,shape[0]*shape[1]*shape[2]
                print( counter + 1 )
                A[x,y,z] = counter + 1
    
    print("Are the two arrays the same now ?")
    same_check = ( A == my_3D_array ).all() 
    # Checks to make sure all the elements are the same. 
    # For integers, it is okay.
    print( f"A == my_3D_array --> {A == my_3D_array}" )
    print( f"\n(A == my_3D_array).all() --> {(A == my_3D_array).all()}" )
    print("\n What about floats ?")
    print("Let's make a random array of VERY small numbers:")
    random_array = np.array([ [ [random.random()*10**-10    \
                                 for z in range(shape[2])]  \
                                 for y in range(shape[1]) ] \
                                 for x in range(shape[0]) ])
    print( random_array )
    print( f"\n(A + rand == my_3D_array).all() --> {(A + random_array == my_3D_array).all()}" )
    print(f"\nCompare floats with built-in numpy function: np.allclose()")
    print( f"np.allclose(A + rand, my_3D_array) --> {np.allclose(A + random_array, my_3D_array)}" )

    print("\n####### END MAKE AND COMPARE ARRAYS #######\n")

def learn_array_math():
    """
    Here, we will learn about simple mathematical operations on numpy arrays.
    """

    print("\n####### BEGIN ARRAY MATH #######\n")

    A = np.array([1,2,3,4,5])
    print(f"A      = {A}")
    
    print("\nArrays with numbers:")
    print(f"A + 1  = {A + 1} = A + np.array([1,1,1,1,1]) = {A + np.array([1,1,1,1,1])}")
    print(f"A - 1  = {A - 1}")
    print(f"A * 2  = {A * 2}")
    print(f"A / 2  = {A / 2}")
    print(f"A // 2 = {A // 2}")

    print("\nArrays with arrays:")
    A = np.array([1,2,3,4,5])
    B = np.array([5,5,5])
    print(f"A      = {A}")
    print(f"B      = {B}")
    print(f"A + B --> DOES NOT WORK ! Shapes need to be the same for all operations.\n")

    A = np.array([1,2,3,4,5])
    B = np.array([5,5,5,5,5])
    print(f"A      = {A}")
    print(f"B      = {B}")
    print(f"A + B  = {A + B}")
    print(f"A - B  = {A - B}")
    print(f"A * B  = {A * B}")
    print(f"A / B  = {A / B}")

    print("\nOften in QM, one encounters the needs to get differences between\n\
    sets of eigenvalues (i.e., energies) of wavefunctions as matrices.")
    ENERGY = np.array([1,2,3,4,5])
    E_DIFF = np.subtract.outer(ENERGY,ENERGY)
    print("E_DIFF = \n",E_DIFF)
    print("\nSometimes, one needs to have 1/E_DIFF (e.g., non-adiabatic coupling = d_jk ~ 1/E_DIFF for j != k):\n")
    print("1/E_DIFF = \n",1/E_DIFF)
    print("\nWe found 'inf' terms since we divided by 0.")
    print("We should have set the diagonal elements to 1.0 before inverting. Then set back to 0.0.")
    E_DIFF[ np.diag_indices(len(ENERGY)) ] = 1.0 # Set diagonal elements to one before inverting.
    E_DIFF_INV = 1/E_DIFF
    E_DIFF_INV[ np.diag_indices(len(ENERGY)) ] = 0.0 # Return indices to zero.
    print("1/E_DIFF = \n",E_DIFF_INV)
    print("\nWe will make use of this stuff once we get to quantum mechanics.")

    print("\nArray with complicated math:")
    A = np.array([1,2,3,4,5])
    print(f"A      = {A}")
    print(f"e^A    = {np.exp(A)}")
    print("\t-->This raised each individual element into the power of e.")
    print(f"LOG[A]    = {np.log(A)}")
    print("\t-->This took the natural log of each individual element.")
    print("\nWhat about matrices ? Does EXP[M] make sense the way it works with vectors ?")
    A = np.array([[1,2],[2,5]])
    print(f"A         = \n{A}")
    print(f"np.exp()  = \n{np.exp(A)}")
    print("\t--> This does not make sense ! How do we do exponentials of matrices ? \
                \n\t\t--> Open mathematical question !")
    print("\t--> Recall that exponentials of matrices are very common in QM: \
                \n\t\tU|\psi(0)> ~ e^(HAMILTONIAN*t)|\psi(t)>")
    print("e^A = 1 + 1/2 A^2 + 1/6 A^3 + 1/24 A^4 + ... (Taylor expansion)")
    print("We can either (1) (i) diagonalize the matrix, \
                \n\t\t(ii) exponentiate the diagonals (math is okay if matrix is digonal), \
                \n\t\tthen (iii) rotate back, OR,")
    print("\t(2) we can use the Pade approximation.")
    print("\n(1) Rotation to Diagonal Space")
    Av, U = np.linalg.eigh(A)
    print(f" U @ np.diag(Av) @ U.T = \n{U @ np.diag(Av) @ U.T}")
    print(f" U @ np.diag(np.exp(Av)) @ U.T = \n{U @ np.diag(np.exp(Av)) @ U.T}")

    print("\n(2) Pade Approximation")
    print(f"SCIPY_MAT_EXP(A) = \n{SCIPY_MAT_EXP(A)}" )

    print("\tYou may ask: if we know two ways to do it, why is it an open question ?")
    print("\tThe problem is that in QM, the matrices are HUGE. Here, we did 2x2.")
    print("\tFor realistic problems, one may find matrices of NxN with N ~ 10^50")
    print("\tThe Pade approximation is much faster than (i) diagonalization, (ii) exponentiation, (iii) rotation")

    print("\nNote: '@' symbolizes matrix multiplication with 2D numpy arrays.")
    print("We will discuss this more when we get to linear algebra chapter.")

    #print("\n####### END ARRAY MATH #######\n")

def main():
    learn_make_numpy_array()
    #learn_array_math()

if ( __name__ == "__main__" ):
    main()