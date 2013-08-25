#!/usr/bin/env python

import numpy as np
import pyviennacl as p
import random

A_layout = p.COL_MAJOR
B_layout = p.COL_MAJOR
C_layout = p.COL_MAJOR

size1 = 3
size2 = 3
size3 = 3

dtype = p.float64

if A_layout == p.ROW_MAJOR:
    A_order = 'C'
else:
    A_order = 'C'

if B_layout == p.ROW_MAJOR:
    B_order = 'C'
else:
    B_order = 'F'

if C_layout == p.ROW_MAJOR:
    C_order = 'C'
else:
    C_order = 'F'

value = random.random()

# Create reference numpy types
A = np.ones((size1, size2), dtype = dtype, order = A_order) * value
big_A = np.ones((size1 * 4, size2 * 4), dtype = dtype, order = A_order) * value
B = np.ones((size2, size3), dtype = dtype, order = B_order) * value
big_B = np.ones((size2 * 4, size3 * 4), dtype = dtype, order = B_order) * value
C = np.ones((size1, size3), dtype = dtype, order = C_order) * value
big_C = np.ones((size1 * 4, size3 * 4), dtype = dtype, order = C_order) * value

A_trans = A.T
big_A_trans = big_A.T
B_trans = B.T
big_B_trans = big_B.T
C_trans = C.T
big_C_trans = big_C.T

# Construct appropriate ViennaCL objects
vcl_A = p.Matrix(A, layout = A_layout)

vcl_big_range_A = p.Matrix(big_A, layout = A_layout)
vcl_range_A = vcl_big_range_A[size1:2*size1, size2:2*size2]

print(p._viennacl.matrix_col_double(big_A).as_ndarray())

vcl_big_slice_A = p.Matrix(big_A, layout = A_layout)
vcl_slice_A = vcl_big_slice_A[size1:-size1:2, size2::3]

A_range = big_A[size1:2*size1, size2:2*size2]

# In-place add
old_vcl_range_A = vcl_range_A.value
old_shape = vcl_range_A.shape
old_A_range = A_range.copy()

A_range += A_range
vcl_range_A += vcl_range_A

if not (vcl_range_A == A_range).all():
    print(old_shape, vcl_range_A.shape)
    print("old_vcl_range_A", old_vcl_range_A)
    print("old A_range", old_A_range)
    print("new vcl_range_A", vcl_range_A.value)
    print("new A_range", A_range)
    print("diff", vcl_range_A.value - A_range)
    print("vcl_range_A flags", vcl_range_A.value.flags)
    raise RuntimeError("Failed: in-place add")

