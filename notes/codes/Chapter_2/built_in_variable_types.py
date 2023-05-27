# NAME: built_in_variable_types.py

import math
import numpy as np

def learn_integers():
    """
    In this function, we will learn about integers.
    """

    print("\n####### BEGIN INTEGERS #######\n")

    a = 10 # Store the integer value of "10" in variable "a"
           # Python will automatically know that "10" is an integer
    # To check this, use the built-in type() function
    print(f"What type is variable a ?  --> {type(a)}" )

    # What operations can we do on integers ?
    # Addition
    print(f"a = {a}  -->  a + 1 = {a + 1}    --> {type(a + 1)}" )
    # Subtraction
    print(f"a = {a}  -->  a - 1 = {a - 1}     --> {type(a - 1)}" )
    # Multiplication
    print(f"a = {a}  -->  a*5   = {a * 5}    --> {type(a * 5)}" )
    # Division
    print(f"a = {a}  -->  a/5   = {a / 5}   --> {type(a / 5)}" )
    print("  Warning! Division produced a floating point variable.")
    print("  To achieve an integer again, we need to do integer division.")
    print("  Note that if the floating point result is 1.75, int(1.75) = 1 (i.e., rounds down)")
    # Integer Division
    print(f"a = {a}  -->  a//5      = {a // 5}         --> {type(a // 5)}" )
    print(f"a = {a}  -->  int(a/5)  = {int(a / 5)}     --> {type(int(a / 5))}" )
    print()

    # Integer Power
    print(f"a = {a}  -->  a**2   = {a ** 2}    --> {type(a ** 2)}" )
    print()

    # Other operations produce floating points
    # Fractional Power (e.g., a square-root)
    print("The following types of operations necessarily produce non-integer results:")
    print(f"a = {a}  -->  a**(0.5)       = {a ** (0.5)}    --> {type(a ** (0.5))}" )
    print(f"a = {a}  -->  a**(1/2)       = {a ** (1/2)}    --> {type(a ** (1/2))}" )
    print(f"a = {a}  -->  math.sqrt(a)   = {math.sqrt(a)}    --> {type(math.sqrt(a))}" )
    print(f"a = {a}  -->  math.log[a]    = {math.log(a)}     --> {type(math.log(a))}" )

    print("\n####### END INTEGERS #######\n")


def learn_floating_points():
    """
    In this function, we will learn about floating point variables.
    """
    print("\n####### BEGIN FLOATS #######\n")

    a = 2.8 # Store the floating point value of "2.8" in variable "a"
            # Python will automatically know that "2.8" is an float
    # To check this, use the built-in type() function
    print(f"What type is variable a ?  --> {type(a)}" )

    # What operations can we do on integers ?
    # Addition
    print(f"a = {a}  -->  a + 1 = {a + 1}     --> {type(a + 1)}" )
    # Subtraction
    print(f"a = {a}  -->  a - 1 = {round(a - 1,3)}     --> {type(a - 1)}" )
    # Multiplication
    print(f"a = {a}  -->  a*5   = {a * 5}    --> {type(a * 5)}" )
    # Division
    print(f"a = {a}  -->  a/5   = {round(a / 5,3)}    --> {type(a / 5)}" )
    
    print("\nWhat if we need an integer from a floating point ?")
    # Convert Result to Integer
    print(f"a = {a}  -->  int(a)         = {int(a)}     --> {type(int(a))}" )
    print(f"a = {a}  -->  math.floor(a)  = {math.floor(a)}     --> {type(math.floor(a))}" )
    print(f"a = {a}  -->  math.ceil(a)   = {math.ceil(a)}     --> {type(math.ceil(a))}" )

    print("\nWhat if we only want the first few decimal places ?")
    a = math.pi
    print(f"a = {a}  -->  round(a,0) = {round(a,0)} ")
    print(f"a = {a}  -->  round(a,1) = {round(a,1)} ")
    print(f"a = {a}  -->  round(a,2) = {round(a,2)} ")
    print(f"a = {a}  -->  round(a,3) = {round(a,3)} ")

    # Other operations work as expected
    print("\nThe other operations work as expected.")
    print(f"a = {a}  -->  a**(2)         = {a ** (2)}     --> {type(a ** (2))}" )
    print(f"a = {a}  -->  a**(0.5)       = {a ** (0.5)}    --> {type(a ** (0.5))}" )
    print(f"a = {a}  -->  a**(1/2)       = {a ** (1/2)}    --> {type(a ** (1/2))}" )
    print(f"a = {a}  -->  math.sqrt(a)   = {math.sqrt(a)}    --> {type(math.sqrt(a))}" )
    print(f"a = {a}  -->  math.log[a]    = {math.log(a)}    --> {type(math.log(a))}" )

    print("\n####### END FLOATS #######\n")

def learn_complex_floats():
    """
    In this function, we will learn about complex variables.
    """
    print("\n####### BEGIN COMPLEX #######\n")

    # Two ways to make a complex
    a = complex(2,3)
    a = 2 + 3j # Store the integer value of "2 + 3j" in variable "a"
          # Python will automatically know that "2 + 3j" is complex
    # To check this, use the built-in type() function
    print(f"What type is variable a ?  --> {type(a)}" )

    # What operations can we do on integers ?
    # Addition
    print(f"a = {a}  -->  a + 1  = {a + 1}        --> {type(a + 1)}" )
    print(f"a = {a}  -->  a + 1j = {a + 1j}        --> {type(a + 1j)}" )
    # Subtraction
    print(f"a = {a}  -->  a - 1  = {a - 1}        --> {type(a - 1)}" )
    print(f"a = {a}  -->  a - 1j = {a - 1j}        --> {type(a - 1j)}" )
    # Multiplication
    print(f"a = {a}  -->  a*5    = {a * 5}      --> {type(a * 5)}" )
    print(f"a = {a}  -->  a*5j   = {a * 5j}     --> {type(a * 5j)}" )
    # Division
    print(f"a = {a}  -->  a/5    = {a / 5}    --> {type(a / 5)}" )
    print(f"a = {a}  -->  a/5j   = {a / 5j}    --> {type(a / 5j)}" )
    
    print("\nWhat if we only want the first few decimal places ?")
    print("Here we can invoke Numpy to do this easily for us.")
    a = math.pi + math.pi/2 * 1.0j
    print(f"a = {a}  -->  np.around(a,0) = {np.around(a,0)} ")
    print(f"a = {a}  -->  np.around(a,1) = {np.around(a,1)} ")
    print(f"a = {a}  -->  np.around(a,2) = {np.around(a,2)} ")
    print(f"a = {a}  -->  np.around(a,3) = {np.around(a,3)} ")

    # Other operations work as expected
    print("\nThe other operations work as expected.")
    print(f"a = {a}  -->  a**(2)         = {a ** (2)}   ")
    print(f"a = {a}  -->  a**(0.5)       = {a ** (0.5)} ")
    print(f"a = {a}  -->  a**(1/2)       = {a ** (1/2)} ")
    try:
        print(f"a = {a}  -->  math.sqrt(a)   = {math.sqrt(a)} ")
    except TypeError:
        print("math.sqrt(a) didn't work because a is complex.")
    try:
        print(f"a = {a}  -->  math.log[a]    = {math.log(a)} ")
    except TypeError:
        print("math.log[a] didn't work because a is complex.")
    

    # Complex numbers have other operations
    # Complex magnitude = |a| = sqrt[ RE**2 + IM**2 ]
    # Complex phase     = |a| = arctan [ IM / RE ]
    rad_to_deg = 180/math.pi

    my_magnitude     = math.sqrt(a.real**2 + a.imag**2)
    numpy_magnitude  = np.abs(a)
    
    my_angle         = math.atan(a.imag / a.real)
    numpy_angle      = np.angle(a, deg=True) # Result in degrees rather than radians
    print("\nComplex Magnitude (degrees):", round( my_magnitude,4), round(numpy_magnitude,4) )
    print("Complex Angle     (degrees):", round( my_angle * rad_to_deg,4), round(numpy_angle,4) )

    print("\n####### END COMPLEX #######\n")

def learn_strings():
    """
    The purpose of this function is to learn strings.
    """

    print("\n####### BEGIN STRING #######\n")

    # A string is a list of characters
    my_string = "Hello, user"
    print(f"My String: '{my_string}' has {len(my_string)} characters. Yes, spaces count here.")
    print(f"The eighth character is my string is '{my_string[7]}' ")
    print(f"\nWhat if we try mathematical operations on strings ? Does it make sense ?")
    print(f" my_string + my_string = '{my_string + my_string}'  --> It became a longer string.")
    print(f" my_string * 2         = '{my_string * 2}'  --> It became a longer string.")
    
    try:
        print(f"my_string + 2 = '{my_string + 2}'")
    except TypeError:
        print("\tmy_string + 2 does not work")
    try:
        print(f"my_string / 2 = '{my_string / 2}'")
    except TypeError:
        print("\tmy_string / 2 does not work")


    # We can check if things are in the string
    print(f"\n Is 'user' in my string ?    --> {'user' in my_string}")
    print(f" Is 'teacher' in my string ? --> {'teacher' in my_string}")

    # Check if two strings are equal
    print(''' Is "string_5" equal to "string_5" ? -->''', "string_5" == "string_5" )
    print(''' Is "string_5" equal to "not the same string" ? -->''', "string_5" == "not the same string" )

    # We can split strings based on characters
    print(f"\n my_string.split() splits based on spaces by default --> {my_string.split()} ")
    print(f" my_string.split(' ') splits based on spaces --> {my_string.split(' ')} ")
    print(f" my_string.split(',') splits based on commas --> {my_string.split(',')} ")
    print(f" my_string.split('ll') splits based on 'll' --> {my_string.split('ll')} ")

    print(" The anove variable types are called lists. We will see this later.")

    print("\n####### END STRING #######\n")

def learn_booleans():
    """
    The purpose of this function is to learn boolean variables.
    """

    print("\n####### BEGIN BOOL #######\n")

    print("Boolean variables are True or False.")

    my_true_bool  = True  # First letter MUST be capitalized. true will give an error.
    my_false_bool = False # First letter MUST be capitalized. false will give an error.

    print(f"Is True equal to False ? --> {my_true_bool == my_false_bool}")
    print(f"Is True equal to True  ? --> {my_true_bool == my_true_bool}")

    print("\nWhat about mathematical operations, again ?")
    print(f" True  * 0 = {True * 0}")
    print(f" True  * 1 = {True * 1}")
    print(f" True  * 5 = {True * 5}")
    print(f" False * 5 = {False * 5}")
    print(" I use this feature quite often, even though it is strange.")
    print(" It allows quickly choosing between variables given conditions.")
    print(" We will use this later in the course.")
    print(f" 5 * (1 == 2) + 10 * (2 == 2) = {5 * (1 == 2) + 10 * (2 == 2)}")

    print("\n####### END BOOL #######\n")

def learn_lists():
    """
    The purpose of this function is to learn about lists.
    """

    print("\n####### BEGIN LIST #######\n")

    my_list = [ "Hello", 5, 2.5, True, [1,2.,"3",4.5] ]
    print(" Lists are collections of variables.")
    print(f''' "[ 'Hello', 5, 2.5, True, [1,2,3,4] ]" = {my_list}''')
    print(" They can store the same type or different types of variables within them.")
    print(" Get elements from the list:")
    print(f" my_list    = {my_list} --> {type(my_list)} ")
    print(f" my_list[0] = {my_list[0]} --> {type(my_list[0])} ")
    print(f" my_list[1] = {my_list[1]}     --> {type(my_list[1])} ")
    print(f" my_list[2] = {my_list[2]}   --> {type(my_list[2])} ")
    print(f" my_list[3] = {my_list[3]}  --> {type(my_list[3])} ")
    print(f" my_list[4] = {my_list[4]}  --> {type(my_list[4])} ")
    print(" The last element is actually another list filled with integers.")
    print(" To access these 'deeper' elements, we need to access them via a second index:")
    print(f" my_list[4][0] = {my_list[4][0]}  --> {type(my_list[4][0])} ")
    print(f" my_list[4][1] = {my_list[4][1]}  --> {type(my_list[4][1])} ")
    print(f" my_list[4][2] = {my_list[4][2]}  --> {type(my_list[4][2])} ")
    print(f" my_list[4][3] = {my_list[4][3]}  --> {type(my_list[4][3])} ")
    print("\n Lists are very powerful for storing data, but their mathemetical operations are limited:")
    print(f" my_list + my_list = {my_list + my_list} --> Longer List")
    print(f" my_list * 2 = {my_list * 2} --> Longer List")
    print(" They act like strings actually.")

    print("\n Another nice feature of lists is that you can dynamically change them:")
    my_list = []
    print(f"my_list = {my_list}")
    my_list.append("1")
    print(f"my_list = {my_list}")
    my_list.append("2")
    print(f"my_list = {my_list}")
    my_list.append("3")
    print(f"my_list = {my_list}")
    print("This is useful for when you don't know the number of elements you will need.")


    print("\n####### END LIST #######\n")

def learn_dictionaries():
    """
    The purpose of this function is to learn about dictionaries.
    """

    print("\n####### BEGIN DICTIONARY #######\n")

    print(" Dictionaries store keys and values and are able to be referenced by the keys.")
    my_dict = { "key":"value", 10:"ten", "carbon":12.007, True:"I am true." }
    print( my_dict )
    print( f" my_dict['key'] = {my_dict['key']}" )
    print( f" my_dict[10] = {my_dict[10]}" )
    print( f" my_dict['carbon'] = {my_dict['carbon']}" )
    print( f" my_dict[True] = {my_dict[True]}" )
    print(" One can add keys and values dynamically:")
    my_dict["NEW_KEY"] = "NEW_VALUE"
    print( f" my_dict['NEW_KEY'] = {my_dict['NEW_KEY']}" )


    print("\n####### END DICTIONARY #######\n")



def main():
    learn_integers()
    #learn_floating_points()
    #learn_complex_floats()
    #learn_strings()
    #learn_booleans()
    #learn_lists()
    #learn_dictionaries()


if ( __name__ == "__main__" ):
    main()