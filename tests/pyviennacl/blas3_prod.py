#!/usr/bin/env python

import math
import os
import pyviennacl as p
import sys

from test_common import diff, test_matrix_layout


def run_test(*args, **kwargs):
    """
    A, A_trans, B, B_trans must be numpy array or matrix instances
    """
    epsilon = args[0]
    A = args[1]
    A_trans = args[2]
    B = args[3]
    B_trans = args[4]
    C = args[5]
    vcl_A = args[6]
    vcl_A_trans = args[7]
    vcl_B = args[8]
    vcl_B_trans = args[9]
    vcl_C = args[10]

    # C +-= A * B
    C = A.dot(B)
    vcl_C = vcl_A * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = A * B passed!")

    C += A.dot(B)
    vcl_C += vcl_A * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += A * B passed!")

    C -= A.dot(B)
    vcl_C -= vcl_A * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= A * B passed!")

    # C +-= A * trans(B)
    C = A.dot(B_trans.T)
    vcl_C = vcl_A * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = A * trans(B) passed!")

    C += A.dot(B_trans.T)
    vcl_C += vcl_A * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += A * trans(B) passed!")

    C -= A.dot(B_trans.T)
    vcl_C -= vcl_A * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= A * trans(B) passed!")

    # C +-= trans(A) * B
    C = A_trans.T.dot(B)
    vcl_C = vcl_A_trans.T * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = trans(A) * B passed!")

    C += A_trans.T.dot(B)
    vcl_C += vcl_A_trans.T * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += trans(A) * B passed!")

    C -= A_trans.T.dot(B)
    vcl_C -= vcl_A_trans.T * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= trans(A) * B passed!")

    # C +-= trans(A) * trans(B)
    C = A_trans.T.dot(B_trans.T)
    vcl_C = vcl_A_trans.T * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = trans(A) * trans(B) passed!")

    C += A_trans.T.dot(B_trans.T)
    vcl_C += vcl_A_trans.T * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += trans(A) * trans(B) passed!")

    C -= A_trans.T.dot(B_trans.T)
    vcl_C -= vcl_A_trans.T * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= trans(A) * trans(B) passed!")

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

    print("*** Using float numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-3
    print("  eps:      %s" % epsilon)
    test_matrix_layout(run_test, epsilon, p.float32)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-11
    print("  eps:      %s" % epsilon)
    test_matrix_layout(run_test, epsilon, p.float64)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())

