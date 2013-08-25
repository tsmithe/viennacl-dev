#!/usr/bin/env python

import math
import numpy as np
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

    dtype = kwargs['dtype']

    alpha = p.Scalar(dtype(3.1415))
    beta = p.HostScalar(dtype(2.718))

    # Test initialisers
    # + GPU scalar TODO
    #X = p.Matrix(A.shape, alpha)
    #if not (X == (np.ones(A.shape, dtype = dtype) * alpha.value)).all():
    #    raise RuntimeError("Failed: GPU scalar matrix init")

    # + CPU scalar TODO
    Y = p.Matrix(A.shape, beta.value) # TODO
    if not (Y == (np.ones(A.shape, dtype = dtype) * beta.value)).all():
        raise RuntimeError("Failed: CPU scalar matrix init")

    # + ndarray
    X = p.Matrix(np.ones(A.shape, dtype = dtype) * beta.value)
    if not (X == (np.ones(A.shape, dtype = dtype) * beta.value)).all():
        raise RuntimeError("Failed: ndarray matrix init")

    # + Matrix
    X = p.Matrix(Y)
    if not (X == Y).all():
        raise RuntimeError("Failed: Matrix Matrix init")

    # + CompressedMatrix
    Y = p.CompressedMatrix(X)
    X = p.Matrix(Y)
    if not (X == Y).all():
        raise RuntimeError("Failed: Matrix CompressedMatrix init")
    
    # In-place add
    X = vcl_A.value
    X += vcl_B.value
    vcl_A += vcl_B
    if not (vcl_A == X).all():
        print("VCL_A", vcl_A.value)
        print("X", X)
        print(vcl_A == X)
        raise RuntimeError("Failed: in-place add")

    # Scaled in-place add
    X += vcl_B.value * alpha.value
    vcl_A += vcl_B * alpha
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: scaled in-place add")

    # Add
    Y = vcl_A.value + vcl_B.value
    Z = vcl_A + vcl_B
    if not (Y == Z).all():
        print(Y)
        print(Z)
        print(Z == Y)
        raise RuntimeError("Failed: add")

    # Scaled add (left)
    Y = dtype(alpha.value) * vcl_B.value + vcl_C.value
    Z = alpha * vcl_B + vcl_C
    act_diff = math.fabs(diff(Y, Z))
    #if not np.allclose(Y, Z, epsilon): # (Z == Y).all():
    if act_diff > epsilon:
        print("Y", Y)
        print("Z", Z)
        print(Z == Y)
        print(act_diff)
        raise RuntimeError("Failed: scaled add (left)")

    # Scaled add (right)
    #X = vcl_A.value
    #Y = X + alpha.value * X
    #Z = (vcl_A + alpha * vcl_A).as_ndarray()
    #if not np.allclose(Y, Z, epsilon): # (Z == Y).all():
    #    print(Y)
    #    print(Z)
    #    print(Z == Y)
    #    raise RuntimeError("Failed: scaled add (left)")

    # Scaled add (both)
    Y = alpha.value * vcl_B.value + beta.value * vcl_C.value
    Z = (alpha * vcl_B + beta * vcl_C).as_ndarray()
    if not np.allclose(Y, Z, epsilon):
        raise RuntimeError("Failed: scaled add (both)")

    # In-place sub
    # Scaled in-place sub
    # Sub
    # Scaled sub (left)
    # Scaled sub (right)
    # Scaled sub (both)

    # Scalar multiplication (CPU scalar)
    # Scalar multiplication (GPU scalar)

    # Matrix-vector multiplication

    # Binary elementwise operations
    # Unary elementwise operations

    # C +-= A * B
    #C = A.dot(B)
    #vcl_C = vcl_A * vcl_B
    #act_diff = math.fabs(diff(C, vcl_C))
    #if (act_diff > epsilon):
    #    raise Exception("Error at operation: matrix-matrix product; diff = %s"
    #                    % act_diff)
    #print("Test C = A * B passed!")

    #C += A.dot(B)
    #vcl_C += vcl_A * vcl_B
    #act_diff = math.fabs(diff(C, vcl_C))
    #if (act_diff > epsilon):
    #    raise Exception("Error at operation: matrix-matrix product; diff = %s"
    #                    % act_diff)
    #print("Test C += A * B passed!")

    #C -= A.dot(B)
    #vcl_C -= vcl_A * vcl_B
    #act_diff = math.fabs(diff(C, vcl_C))
    #if (act_diff > epsilon):
    #    raise Exception("Error at operation: matrix-matrix product; diff = %s"
    #                    % act_diff)
    #print("Test C -= A * B passed!")

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
    #test_matrix_layout(run_test, epsilon, p.float32)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-11
    print("  eps:      %s" % epsilon)
    test_matrix_layout(run_test, epsilon, p.float64, 11, 11, 11)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())

