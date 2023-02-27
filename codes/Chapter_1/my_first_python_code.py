# my_first_python_code.py

# Python modules
import random

# External modules
import numpy as np 
from matplotlib import pyplot as plt 

# User-made modules
# We don't yet have any of these.

# THIS IS A SINGLE-LINE COMMENT
"""
THIS IS A BLOCK COMMENT.
IT CAN GO FOR MANY LINES.
"""

""" Description of the above imports:
  `as np' renames the numpy module to 'np'
  A function inside the module Matplotlib is called pyplot. 
  We want it, and we rename it as plt
"""

def my_print_function(something_to_print):
    """
    This is a function comment. 
    They are useful whenever your define a new function.
    You should always tell the purpose of the function 
      as well as what is its input and output.

    For example:
    The purpose of this function is to read a string and print the string.
    INPUT: something_to_print [str]
    OUTPUT: None
    """
    print( "My function is printing something:" )
    print( something_to_print )

    # This is a return statement. 
    # For this function, it is optional. Default is ``None''/
    return None

def main():
    # This is the main function of the code.
    # This enables one to get main from both the command line 
    #   as well as from another python module.
    
    something_to_print = "Hello, world." # String variable
    
    print("Print directly:", something_to_print) # Print directly to the screen

    # Write a function to print whatever you provide to it
    my_print_function( something_to_print )

# This checks whether someone is calling this module at the command line directly,
#   or whether it is being called from another python file as a user-defined module
if ( __name__ == "__main__" ): # This will evaluate to True if run from the command line
    main()