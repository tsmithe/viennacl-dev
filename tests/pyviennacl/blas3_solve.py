#!/usr/bin/env python

import math
import os
import pyviennacl as p
import scipy.linalg as sp
import sys

from test_common import diff, test_matrix_layout, test_matrix_solvers, noop

def run_test(*args, **kwargs):
    """
    A, A_trans, B, B_trans must be numpy array or matrix instances
    """

    # So, for plain solvers, we need:
    #  * an upper-triangular matrix A 
    #  * a unit-upper-triangular matrix A
    #  * a lower-triangular matrix A 
    #  * a unit-lower-triangular matrix A
    #  * a vector B
    #  * a matrix B
    #  * transposed versions of the matrices above
    # matrices are square
    # want row and col layout
    # and range / slice for A
    
    # For in-place solvers:
    #  ...???...

    epsilon = args[0]
    A = args[1]
    B = args[2]
    vcl_A = args[3]
    vcl_B = args[4]

    act_diff = math.fabs(diff(A, vcl_A))
    if act_diff > epsilon:
        raise Exception("Error copying A")

    act_diff = math.fabs(diff(B, vcl_B))
    if act_diff > epsilon:
        raise Exception("Error copying B")

    act_diff = math.fabs(diff(A_trans, vcl_A_trans))
    if act_diff > epsilon:
        raise Exception("Error copying A_trans")

    act_diff = math.fabs(diff(B_trans, vcl_B_trans))
    if act_diff > epsilon:
        raise Exception("Error copying B_trans")

    # solve and in-place solve
    # transpositions: A \ B, A^T \ B, A \ B^T, A^T \ B^T
    # tags: lower, unit_lower, upper, unit_upper, cg, bicgstab, gmres

    # A \ B
    vcl_X = p.solve(vcl_A, vcl_B, p.upper_tag())
    X = sp.solve(A, B)
    print(X - vcl_X)

    return os.EX_OK


def test():
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("## Test :: BLAS 3 routines")
    print("----------------------------------------------")
    print("----------------------------------------------")
    print()
    print("----------------------------------------------")
    print("--- Part 1: Testing matrix-matrix products ---")

    #print("*** Using float numeric type ***")
    #print("# Testing setup:")
    #epsilon = 1.0E-3
    #print("  eps:      %s" % epsilon)
    #test_matrix_layout(test_matrix_solvers, run_test, 
    #                   epsilon, p.float32, 11, 11, 11)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-11
    print("  eps:      %s" % epsilon)
    test_matrix_layout(test_matrix_solvers, noop, epsilon, p.float64)
    #test_matrix_layout(test_matrix_solvers, run_test, 
    #                   epsilon, p.float64, 5,5,5)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())

# upper, unit_upper, lower, unit_lower
# trans...
