#!/usr/bin/env python

import math
import numpy as np
import os
import pyviennacl as p
import random


def diff(a, b):
    p.util.backend_finish()
    ret = 0

    # Convert NumPy types to ViennaCL types (they're much more convenient!)
    if isinstance(a, np.ndarray):
        if a.ndim == 1:
            a = p.Vector(a)
        elif a.ndim == 2:
            a = p.Matrix(a)
        else:
            raise TypeError("Something went wrong")
    if isinstance(b, np.ndarray):
        if b.ndim == 1:
            b = p.Vector(b)
        elif b.ndim == 2:
            b = p.Matrix(b)
        else:
            raise TypeError("Something went wrong")

    # The MagicMethods class guarantees that we have some useful facilities
    # (both Node and Leaf are derived from MagicMethods)
    if isinstance(a, p.MagicMethods) and isinstance(b, p.MagicMethods):
        d = p.ElementFabs(a - b)
        cpu_d = d.as_ndarray()
        if len(d.shape) == 1:
            # vector
            for i in range(d.shape[0]):
                act = math.fabs(cpu_d[i])
                if act > ret:
                    ret = act
        elif len(d.shape) == 2:
            # matrix
            for i in range(d.shape[0]):
                for j in range(d.shape[1]): 
                    act = math.fabs(cpu_d[i, j])
                    if act > ret:
                        ret = act
        else:
            raise TypeError("Something went wrong..")
        return ret
    else:
        # We don't have either ndarrays or ViennaCL types so assume plain scalar
        return math.fabs(a - b) / max(math.fabs(a), math.fabs(b))


def test_matrix_slice(test_func,
                      epsilon, dtype,
                      A_layout = p.ROW_MAJOR, B_layout = p.ROW_MAJOR, 
                      C_layout = p.ROW_MAJOR,
                      size1 = 131, size2 = 67, size3 = 73):
    if A_layout == p.ROW_MAJOR:
        A_order = 'C'
    else:
        A_order = 'F'

    if B_layout == p.ROW_MAJOR:
        B_order = 'C'
    else:
        B_order = 'F'

    if C_layout == p.ROW_MAJOR:
        C_order = 'C'
    else:
        C_order = 'F'

    # Create reference numpy types
    A = np.empty((size1, size2), dtype = dtype, order = A_order)
    big_A = np.ones((size1 * 4, size2 * 4), dtype = dtype, order = A_order) * 3.1415
    B = np.empty((size2, size3), dtype = dtype, order = B_order)
    big_B = np.ones((size2 * 4, size3 * 4), dtype = dtype, order = B_order) * 42.0
    C = np.empty((size1, size3), dtype = dtype, order = C_order)
    big_C = np.ones((size1 * 4, size3 * 4), dtype = dtype, order = C_order) * 2.718

    # Fill A and B with random values
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = random.random()
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] = random.random()

    A_trans = A.T
    big_A_trans = big_A.T
    B_trans = B.T
    big_B_trans = big_B.T
    C_trans = C.T
    big_C_trans = big_C.T

    # Construct appropriate ViennaCL objects
    vcl_A = p.Matrix(A, layout = A_layout)

    vcl_big_range_A = p.Matrix(big_A, layout = A_layout)
    vcl_big_range_A[size1:2*size1, size2:2*size2] = vcl_A
    vcl_range_A = vcl_big_range_A[size1:2*size1, size2:2*size2]

    vcl_big_slice_A = p.Matrix(big_A, layout = A_layout)
    vcl_big_slice_A[size1:-size1:2, size2::3] = vcl_A
    vcl_slice_A = vcl_big_slice_A[size1:-size1:2, size2::3]

    vcl_A_trans = p.Matrix(A_trans, layout = A_layout)

    vcl_big_range_A_trans = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_range_A_trans[size2:2*size2, size1:2*size1] = vcl_A_trans
    vcl_range_A_trans = vcl_big_range_A_trans[size2:2*size2, size1:2*size1]

    vcl_big_slice_A_trans = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_slice_A_trans[size2:-size2:2, size1::3] = vcl_A_trans
    vcl_slice_A_trans = vcl_big_slice_A_trans[size2:-size2:2, size1::3]

    vcl_B = p.Matrix(B, layout = B_layout)

    vcl_big_range_B = p.Matrix(big_B, layout = B_layout)
    vcl_big_range_B[size2:2*size2, size3:2*size3] = vcl_B
    vcl_range_B = vcl_big_range_B[size2:2*size2, size3:2*size3]

    vcl_big_slice_B = p.Matrix(big_B, layout = B_layout)
    vcl_big_slice_B[size2:-size2:2, size3::3] = vcl_B
    vcl_slice_B = vcl_big_slice_B[size2:-size2:2, size3::3]

    vcl_B_trans = p.Matrix(B_trans, layout = B_layout)

    vcl_big_range_B_trans = p.Matrix(big_B_trans, layout = B_layout)
    vcl_big_range_B_trans[size3:2*size3, size2:2*size2] = vcl_B_trans
    vcl_range_B_trans = vcl_big_range_B_trans[size3:2*size3, size2:2*size2]

    vcl_big_slice_B_trans = p.Matrix(big_B_trans, layout = B_layout)
    vcl_big_slice_B_trans[size3:-size3:2, size2::3] = vcl_B_trans
    vcl_slice_B_trans = vcl_big_slice_B_trans[size3:-size3:2, size2::3]

    vcl_C = p.Matrix(C, layout = C_layout)

    vcl_big_range_C = p.Matrix(big_C, layout = C_layout)
    vcl_range_C = vcl_big_range_C[(size1 - 1):(2*size1 - 1), (size3 - 1):(2*size3 - 1)]

    vcl_big_slice_C = p.Matrix(big_C, layout = C_layout)
    vcl_slice_C = vcl_big_slice_C[(size1 - 1):(4*size1 - 1):3, (size3 - 1):(4*size3 - 1):3]

    # A=matrix, B=matrix, C=matrix
    print("Now using A=matrix, B=matrix, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_B, vcl_B_trans, vcl_C,
                    dtype = dtype)

    # A=matrix, B=matrix, C=range
    print("Now using A=matrix, B=matrix, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_B, vcl_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=matrix, B=matrix, C=slice
    print("Now using A=matrix, B=matrix, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_B, vcl_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=matrix, B=range, C=matrix
    print("Now using A=matrix, B=range, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_C,
                    dtype = dtype)

    # A=matrix, B=range, C=range
    print("Now using A=matrix, B=range, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=matrix, B=range, C=slice
    print("Now using A=matrix, B=range, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=matrix, B=slice, C=matrix
    print("Now using A=matrix, B=slice, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_C,
                    dtype = dtype)

    # A=matrix, B=slice, C=range
    print("Now using A=matrix, B=slice, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=matrix, B=slice, C=slice
    print("Now using A=matrix, B=slice, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_A, vcl_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=range, B=matrix, C=matrix
    print("Now using A=range, B=matrix, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_B, vcl_B_trans, vcl_C,
                    dtype = dtype)

    # A=range, B=matrix, C=range
    print("Now using A=range, B=matrix, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_B, vcl_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=range, B=matrix, C=slice
    print("Now using A=range, B=matrix, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_B, vcl_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=range, B=range, C=matrix
    print("Now using A=range, B=range, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_C,
                    dtype = dtype)

    # A=range, B=range, C=range
    print("Now using A=range, B=range, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=range, B=range, C=slice
    print("Now using A=range, B=range, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=range, B=slice, C=matrix
    print("Now using A=range, B=slice, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_C,
                    dtype = dtype)

    # A=range, B=slice, C=range
    print("Now using A=range, B=slice, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=range, B=slice, C=slice
    print("Now using A=range, B=slice, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_range_A, vcl_range_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=slice, B=matrix, C=matrix
    print("Now using A=slice, B=matrix, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_B, vcl_B_trans, vcl_C,
                    dtype = dtype)

    # A=slice, B=matrix, C=range
    print("Now using A=slice, B=matrix, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_B, vcl_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=slice, B=matrix, C=slice
    print("Now using A=slice, B=matrix, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_B, vcl_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=slice, B=range, C=matrix
    print("Now using A=slice, B=range, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_C,
                    dtype = dtype)

    # A=slice, B=range, C=range
    print("Now using A=slice, B=range, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=slice, B=range, C=slice
    print("Now using A=slice, B=range, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_range_B, vcl_range_B_trans, vcl_slice_C,
                    dtype = dtype)

    # A=slice, B=slice, C=matrix
    print("Now using A=slice, B=slice, C=matrix")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_C,
                    dtype = dtype)

    # A=slice, B=slice, C=range
    print("Now using A=slice, B=slice, C=range")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_range_C,
                    dtype = dtype)

    # A=slice, B=slice, C=slice
    print("Now using A=slice, B=slice, C=slice")
    ret = test_func(epsilon,
                    A, A_trans, B, B_trans, C,
                    vcl_slice_A, vcl_slice_A_trans,
                    vcl_slice_B, vcl_slice_B_trans, vcl_slice_C,
                    dtype = dtype)

    return os.EX_OK


def test_matrix_layout(test_func, epsilon, dtype,
                       size1 = 131, size2 = 67, size3 = 73):
    # A=row, B=row, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=row, B=row, C=row ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.ROW_MAJOR, p.ROW_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    # A=row, B=row, C=col
    print("///////////////////////////////////////")
    print("/// Now testing A=row, B=row, C=col ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.ROW_MAJOR, p.ROW_MAJOR, p.COL_MAJOR, size1, size2, size3)

    # A=row, B=col, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=row, B=col, C=row ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.ROW_MAJOR, p.COL_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    # A=row, B=col, C=col
    print("///////////////////////////////////////")
    print("/// Now testing A=row, B=col, C=col ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.ROW_MAJOR, p.COL_MAJOR, p.COL_MAJOR, size1, size2, size3)

    # A=col, B=row, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=col, B=row, C=row ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.COL_MAJOR, p.ROW_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    # A=col, B=row, C=col
    print("///////////////////////////////////////")
    print("/// Now testing A=col, B=row, C=col ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.COL_MAJOR, p.ROW_MAJOR, p.COL_MAJOR, size1, size2, size3)

    # A=col, B=col, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=col, B=col, C=row ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.COL_MAJOR, p.COL_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    # A=col, B=col, C=col
    print("///////////////////////////////////////")
    print("/// Now testing A=col, B=col, C=col ///")
    print("///////////////////////////////////////")
    test_matrix_slice(test_func, epsilon, dtype, p.COL_MAJOR, p.COL_MAJOR, p.COL_MAJOR, size1, size2, size3)

    return os.EX_OK
