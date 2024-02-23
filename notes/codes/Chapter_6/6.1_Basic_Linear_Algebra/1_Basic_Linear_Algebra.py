import numpy as np
from matplotlib import pyplot as plt


def vectors():
    v = np.array([ 1, 2, 3 ])
    print( v )

    # Check the length of the vectors
    LENGTH_OF_v = len( v )
    print(f"The length of v is {LENGTH_OF_v}\n")

    # Multiply vector by a scalar (a number)
    c  = 5
    vc = v * c
    print( f"v * c = {vc}\n" )

    # Adding a scalar to a vector
    c = 5
    v_plus_c = v + c
    print( f"v + c = {v_plus_c}\n" )

    # Add two vectors
    v1 = np.array([ 1, 2, 3 ])
    v2 = np.array([ 10, 11, 12 ])
    v3 = v1 + v2
    print( f"v1 + v2 = {v3}\n" )

    ## Multiply two vectors
    # Scalar/Dot/Inner Product
    v1 = np.array([ 1, 2, 3 ])
    v2 = np.array([ 10, 11, 12 ])
    v3 = np.sum( v1 * v2 ) # np.dot( v1, v2 )
    print(v1, v2)
    print( f"v1 .dot. v2 = {v3}\n" )

    # Element-wise Product
    v1 = np.array([ 1, 2, 3 ])
    v2 = np.array([ 10, 11, 12 ])
    v3 = v1 * v2
    print(v1, v2)
    print( f"v1 * v2 = {v3}\n" )

    # Outer Product
    v1 = np.array([ 1, 2, 3 ])
    v2 = np.array([ 10, 11, 12 ])
    v3 = np.outer(v1,  v2)
    print(v1, v2)
    print( f"v1 v2.T = {v3}\n" )

    # Cross Product
    v1 = np.array([ 1, 2, 3 ])
    v2 = np.array([ 1, 1, 1 ])
    v3 = np.cross(v1,  v2)
    print(v1, v2)
    print( f"cross(v1, v2) = \n{v3}" )
    print( "Check orthogonality:", np.dot( v1, v3 ), np.dot( v2, v3 ) )

    # Division of two vectors
    v1 = np.array([ 1, 2, 3 ])
    v2 = np.array([ 6, 6, 6 ])
    v3 = v2 / v1
    print(v1, v2)
    print( f"v2 / v2 = {v3}\n" )

    # Non-linear/transcendental of two vectors
    v  = np.array([ 1, 2, 3 ])
    print(v)
    print( f"v**10 = {v**10}" )
    print( f"v**(-5/6) = {v**(-5/6)}" )
    print( f"sin(v) = {np.sin(v)}" )
    print( f"sqrt(v) = {np.sqrt(v)}" )
    print( f"exp(v) = {np.exp(v)}\n" )

    # Complex vectors
    v  = np.array([ 1, 2, 3 ]) + 0j
    print( v )
    v  = np.array([ 1, 2, 3 ], dtype=complex)
    print( v )

    # Complex conjugation
    v  = np.array([ 1, 2, 3 ]) + 1j * np.array([ 1, 2, 3 ])
    print("Vector:          ",v)
    print("Conjugate Vector:", np.conjugate(v) )

def vectors_einsum():
    v = np.array([ 1, 2, 3 ])

    # Dot/Inner/Scalar product
    print("Dot Product:")
    print( np.dot(v,v) )
    print( np.einsum("a,a->",v,v) ) # \SUM_a v[a] * v[a]

    # Element-wise product
    print("Element-wise Product:")
    print( np.dot(v,v) )
    print( np.einsum("a,a->a",v,v) ) # v[a] * v[a] = v1[a]

    # Outer product
    print("Outer Product:")
    print( np.outer(v,v) )
    print( np.einsum("a,b->ab",v,v) ) # v[a] * v[b] =  M[a,b] 

    # Element-wise Division
    print("Element-wise Division:")
    print( v/v )
    print( np.einsum("a,a->a",v,1/v) ) # v[a] * (1/v[a]) =  v1[a]

def matrices():
    N = 2
    A = np.zeros( (N,N) )
    print("A = \n", A )
    print( "A.shape = ", A.shape )

    # Add a scalar to a matrix
    print( "c + A\n", 5 + A )

    # Multiply a scalar to a matrix
    A = np.ones( (N,N) )
    print( "c * A\n", 10 * A )

    # Divide a scalar to a matrix
    A = np.ones( (N,N) )
    print( "c * A\n", A / 10 )

    # Add a vector to a matrix (should it be possibe ???)
    A = np.ones( (N,N) )
    v = np.array( [1,2] )
    print(A)
    print(v)
    print( "v + A\n", v + A, "\n" )

    # Multiply a vector to a matrix (should it be possibe ???)
    A = np.ones( (N,N) )
    v = np.array( [1,2] )
    print(A)
    print(v)
    print( "v * A\n", v * A, "\n" )

    # Multiply a vector by a matrix (should it be possibe ???)
    A = np.array( [[1,2],[3,4]] )
    v = np.array( [1,2] )
    print(A)
    print(v)
    print( "A @ v\n", A @ v, "\n" )

    # Multiply a vector by a matrix (should it be possibe ???)
    A = np.array( [[1,2],[3,4]] )
    v = np.array( [1,2] )
    print(A)
    print(v)
    print( "v @ A\n", v @ A )

    # Multiply a matrix by a matrix (should it be possibe ???)
    A = np.array( [[1,2],[3,4]] )
    B = np.array( [[1,5],[10,15]] )
    print(A)
    print(B)
    print( "A @ B\n", A @ B )
    print( "B @ A\n", B @ A )

def matrices_einsum():

    # Multiply a vector by a matrix (should it be possibe ???)
    A = np.array( [[1,2],[3,4]] )
    v = np.array( [1,2] )
    print( "A @ v\n", A @ v )
    print( "A @ v\n", np.einsum("ab,b->a", A, v) )

    # Multiply a vector by a matrix (should it be possibe ???)
    A = np.array( [[1,2],[3,4]] )
    v = np.array( [1,2] )
    print( "v @ A\n", v @ A )
    print( "v @ A\n", np.einsum("a,ab->b", v, A) )
    #print( "v @ A\n", np.einsum("ab,a->b", A, v) ) # DON'T DO THIS WAY, BUT IT WORKS

    # Matrix-Matrix multiplication
    A = np.array( [[1,2],[3,4]] )
    print( "A @ A\n", A @ A )
    print( "A @ A\n", np.einsum("ab,bc->ac", A, A) ) # \SUM_b A[a,b] * A[b,c] = M[a,c]

    # Matrix-Matrix element-wise multiplication
    A = np.array( [[1,2],[3,4]] )
    print( "A * A\n", A * A )
    print( "A * A\n", np.einsum("ab,ab->ab", A, A) ) # A[a,b] * A[a,b] = M[a,b]

def eigenvalues_and_eigenvectors():

    # What are the eigenvalues and eigenvectors of a diagonal matrix ?
    A = np.array( [[1,0],[0,2]] )
    print( "Diagonal Matrix A:\n", A )
    EIGENVALUES, EIGENVECTORS = np.linalg.eigh( A )
    for v in range( len(A) ):
        print( f"Eigenvalue {v} = {EIGENVALUES[v]}; Eigenvector {v} = {EIGENVECTORS[:,v]}" )

    # How to diagonalize a Hermitian (kj is complex conjugate of jk) matrix
    A = np.array( [[1,1],[1,1]] )
    print( "\nNon-Diagonal Matrix A:\n", A )
    E_a, U_A = np.linalg.eigh( A ) # MOST IMPORTANT QUANTUM MECHANICS TOOL

    for a in range( len(A) ):
        print( f"Eigenvalue E_{a} = {E_a[a]}; Eigenvector U_{a} = {U_A[:,a]}" )

    # How do I use the result of the diagonalization to diagonalize the matrix ?
    print("\nRotate A to diagonal representation: [E_a] = U_a @ A @ U_a.T")
    print( U_A @ A @ U_A.T )
    print("\nRotate eigenvalue matrix to non-diagonal representation: A = U_a.T @ [a] @ U_a")
    print( U_A.T @ np.diag(E_a) @ U_A )

    # How to write a different matrix in the representation of A
    B = np.array([[1,2],[2,3]])
    print("\nNon-Diagonal Matrix B:\n", B )
    print("\nRotate B representation that diagonalizes A: B (in A_rep) = U_A @ B @ U_A.T")
    print( U_A @ B @ U_A.T )

    print("\n\n")

def functions_of_Matrices():
    
    # Functions of matrices
    A        = np.array( [[1,1],[1,1]] )
    print( "\nNon-Diagonal Matrix A:\n", A )
    E_a, U_a = np.linalg.eigh( A )
    for a in range( len(A) ):
        print( f"Eigenvalue E_{a} = {E_a[a]}; Eigenvector U_{a} = {U_a[:,a]}" )

    # Taylor Series Expansion of Matrix Exponential
    # exp(A) = 1 + A + A^2/2! + A^3/3! + ...
    #        = 1 + A + A@A/2 + A@A@A/6 + ...

    print("Exact result can be obtained by:\n \
    1. diagonalizing A,\n \
    2. exponentiating the eigenvalues,\n \
    3. rotating back to original representation")
    print("\nexp(A) = U.T @ exp(E_a) @ U")
    print( U_a.T @ np.diag( np.exp( E_a ) ) @ U_a ) # EXACT RESULT

    # APPROXIMATE RESULT
    NTERMS = 2 # Number of terms in the Taylor series expansion
    RESULT = np.identity(2)
    for n in range( 1, NTERMS ):
        TMP = np.identity(2)
        for _ in range( n ):
            TMP = TMP @ A
        RESULT += TMP / np.math.factorial(n)
    print( f"\nTaylor Series Expansion up to {NTERMS} terms:" )
    print( RESULT )
    print("Try changing the 'NTERMS' variable in the code to see how the result changes.")

    print("\n")
    
def eigenvalue_Equations():

    # Eigenvalue Equations
    # A |a> = a |a>
    # A is a matrix, |a> is a vector, a is a scalar

    A = np.array([[1,2],[2,1]])
    Ea, Ua, = np.linalg.eigh(A)
    #Ua = Ua * np.sqrt( len(A) )
    print("Matrix A:\n", A)
    for a in range( len(Ea) ):
        print( "Eigenvalue a_%s = %1.3f; Eigenvector Ua_%s = %s" %(a, Ea[a], a, Ua[:,a]) )


    print( "A x |a>:" )
    print( A @ Ua[:,0] )
    print( A @ Ua[:,1] )

    # print( "A x v / a:" )
    # print( A @ Ua[:,0] / Ea[0] )
    # print( A @ Ua[:,1] / Ea[1] )

    vec = np.array([ 1, 5 ])
    print( "| vec >           =", vec )
    c0 = np.dot( vec, Ua[:,0] )
    c1 = np.dot( vec, Ua[:,1] )
    print( "c0 = < vec | a0 > =", c0 )
    print( "c1 = < vec | a1 > =", c1 )
    print( "c0 * | a0 >  +  c1 * | a1 > =", c0 * Ua[:,0] + c1 * Ua[:,1] )
    print( Ua @ vec )
    

    B = np.array([[1,1],[1,2]])
    Eb, Ub, = np.linalg.eigh(B)
    print("Matrix B:\n", B)
    for b in range( len(Eb) ):
        print( "Eigenvalue b_%s = %1.3f; Eigenvector Ub_%s = %s" %(b, Eb[b], b, Ub[:,b]) )


    # [A,B] = AB - BA
    print( "[A,B] = AB - BA:\n", A @ B - B @ A )




def main():
    #vectors()
    #vectors_einsum()
    #matrices()
    #matrices_einsum()
    #eigenvalues_and_eigenvectors()
    #functions_of_Matrices()
    
    
    eigenvalue_Equations()

if ( __name__ == "__main__" ):
    main()