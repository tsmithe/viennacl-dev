#!/usr/bin/env python

import math
import numpy as np
import os
import pyviennacl as p
import sys

from test_common import diff, test_matrix_layout

#TODO: Change print statements to log statements

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
    #print("Test: initialisation of matrix with GPU scalar passed")

    # + CPU scalar TODO
    Y = p.Matrix(A.shape, beta.value) # TODO
    if not (Y == (np.ones(A.shape, dtype = dtype) * beta.value)).all():
        raise RuntimeError("Failed: CPU scalar matrix init")
    print("Test: initialisation of matrix with CPU scalar passed")

    # + ndarray
    X = p.Matrix(np.ones(A.shape, dtype = dtype) * beta.value)
    if not (X == (np.ones(A.shape, dtype = dtype) * beta.value)).all():
        raise RuntimeError("Failed: ndarray matrix init")
    print("Test: initialisation of matrix with ndarray passed")

    # + Matrix
    X = p.Matrix(Y)
    if not (X == Y).all():
        raise RuntimeError("Failed: Matrix Matrix init")
    print("Test: initialisation of matrix with Matrix passed")

    # + CompressedMatrix
    Y = p.CompressedMatrix(X)
    X = p.Matrix(Y)
    if not (X == Y).all():
        raise RuntimeError("Failed: Matrix CompressedMatrix init")
    print("Test: initialisation of matrix with CompressedMatrix passed")
    
    # In-place add
    X = vcl_A.value
    X += vcl_B.value
    vcl_A += vcl_B
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: in-place add")
    print("Test: in-place add passed")

    # Scaled in-place add
    X += alpha.value * vcl_B.value
    vcl_A += alpha * vcl_B
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: scaled in-place add")
    print("Test: scaled in-place add passed")

    # Add
    Y = vcl_A.value + vcl_B.value
    Z = vcl_A + vcl_B
    if not (Y == Z).all():
        raise RuntimeError("Failed: add")
    print("Test: add passed")

    # Scaled add (left)
    Y = dtype(alpha.value) * vcl_B.value + vcl_C.value
    Z = alpha * vcl_B + vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled add (left)")
    print("Test: scaled add (left) passed")

    # Scaled add (right)
    Y = vcl_B.value + dtype(alpha.value) * vcl_C.value
    Z = vcl_B + alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon: # (Z == Y).all():
        raise RuntimeError("Failed: scaled add (left)")
    print("Test: scaled add (right) passed")

    # Scaled add (both)
    Y = alpha.value * vcl_B.value + alpha.value * vcl_C.value
    Z = alpha * vcl_B + alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled add (both)")
    print("Test: scaled add (both) passed")

    # In-place sub
    X = vcl_A.value
    X -= vcl_B.value
    vcl_A -= vcl_B
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: in-place sub")
    print("Test: in-place sub passed")

    # Scaled in-place sub
    X -= alpha.value * vcl_B.value
    vcl_A -= alpha * vcl_B
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: scaled in-place sub")
    print("Test: scaled in-place sub passed")

    # Sub
    Y = vcl_A.value - vcl_B.value
    Z = vcl_A - vcl_B
    if not (Y == Z).all():
        raise RuntimeError("Failed: sub")
    print("Test: sub passed")

    # Scaled sub (left)
    Y = alpha.value * vcl_B.value - vcl_C.value
    Z = alpha * vcl_B - vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled sub (left)")
    print("Test: scaled sub (left) passed")

    # Scaled sub (right)
    Y = vcl_B.value - alpha.value * vcl_C.value
    Z = vcl_B - alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled sub (right)")
    print("Test: scaled sub (right) passed")

    # Scaled sub (both)
    Y = alpha.value * vcl_B.value - alpha.value * vcl_C.value
    Z = alpha * vcl_B - alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled sub (both)")
    print("Test: scaled sub (both) passed")

    # Scalar multiplication (CPU scalar) -- not supported yet
    #gamma_py = beta.value * beta.value
    #gamma_vcl = beta * beta
    # ...
    # Scalar multiplication (GPU scalar)

    # Matrix-vector multiplication
    vec = p.Vector(vcl_A.shape[0], 3.1415, dtype = dtype)
    X = vcl_A * vec
    Y = vcl_A.value.dot(vec.value)
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: matrix-vector multiplication")
    print("Test: matrix-vector multiplication passed")

    # Matrix divided by scalar
    X = vcl_A.value / alpha.value
    Y = vcl_A / alpha
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: matrix-scalar division")
    print("Test: matrix-scalar division passed")

    # Binary elementwise operations -- prod and div
    X = vcl_A.value * vcl_B.value
    Y = p.ElementProd(vcl_A, vcl_B)
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise matrix-matrix multiplication")
    print("Test: elementwise matrix-matrix multiplication passed")

    X = vcl_A.value / vcl_B.value
    Y = p.ElementDiv(vcl_A, vcl_B)
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise matrix-matrix division")
    print("Test: elementwise matrix-matrix division passed")

    # Unary elementwise operations
    # - abs TODO
    #X = abs(vcl_A.value)
    #Y = p.ElementAbs(vcl_A)
    #act_diff = math.fabs(diff(X, Y))
    #if act_diff > epsilon:
    #    raise RuntimeError("Failed: elementwise abs")
    #print("Test: elementwise abs passed")

    # - acos
    X = np.arccos(vcl_A.value)
    Y = p.ElementAcos(vcl_A).result # TODO THIS SHOULDN'T BE REQUIRED
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise acos")
    print("Test: elementwise acos passed")

    # - asin
    X = np.arcsin(vcl_A.value)
    Y = p.ElementAsin(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise asin")
    print("Test: elementwise asin passed")

    # - atan
    X = np.arctan(vcl_A.value)
    Y = p.ElementAtan(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise atan")
    print("Test: elementwise atan passed")

    # - ceil
    X = np.ceil(vcl_A.value)
    Y = p.ElementCeil(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise ceil")
    print("Test: elementwise ceil passed")

    # - cos
    # - cosh
    # - exp
    # - fabs
    # - floor
    # - log
    # - log10
    # - sin
    # - sinh
    # - sqrt
    # - tan
    # - tanh
    # - trans
    # - norm1
    # - norm2
    # - norm_inf

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

