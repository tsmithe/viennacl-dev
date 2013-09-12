#!/usr/bin/env python

import math
import os
import pyviennacl as p
import scipy.linalg as sp
import sys

from test_common import diff, test_matrix_layout, test_matrix_solvers, noop

def test_kernel(*args, **kwargs):
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
    A_upper, A_unit_upper, A_lower, A_unit_lower, A_trans_upper, A_trans_unit_upper, A_trans_lower, A_trans_unit_lower = args[1]
    B, B_trans = args[2]
    vcl_A_upper, vcl_A_unit_upper, vcl_A_lower, vcl_A_unit_lower, vcl_A_trans_upper, vcl_A_trans_unit_upper, vcl_A_trans_lower, vcl_A_trans_unit_lower = args[3]
    vcl_B, vcl_B_trans = args[4]

    # solve and in-place solve
    # transpositions: A \ B, A^T \ B, A \ B^T, A^T \ B^T
    # tags: lower, unit_lower, upper, unit_upper, cg, bicgstab, gmres

    # A \ B
    vcl_X = p.solve(vcl_A_upper, vcl_B, p.upper_tag())
    X = sp.solve(A_upper, B)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B for upper triangular A")

    return os.EX_OK


def test():
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("## Test :: BLAS 3 routines :: solvers")
    print("----------------------------------------------")
    print("----------------------------------------------")
    print()
    print("----------------------------------------------")

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
    test_matrix_layout(test_matrix_solvers, test_kernel, epsilon, p.float64,
                       5, 5, 5,
                       num_matrices = 2)
    #test_matrix_layout(test_matrix_solvers, run_test, 
    #                   epsilon, p.float64, 5,5,5)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())

# upper, unit_upper, lower, unit_lower
# trans...
