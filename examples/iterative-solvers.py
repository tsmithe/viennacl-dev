#!python

"""
PyViennaCL provides iterative solvers for various dense linear systems.
The API is documented in ``help(pyviennacl.linalg.solve)``. In particular,
the solver to use is determined by the tag instance supplied to the ``solve``
function.

The iterative solvers have various parameters for tuning the error tolerance,
and various requirements for the form of the system matrix, as described in the
documentation for the corresponding tag classes.

For this reason, we only demonstrate here the use of the GMRES solver for an
general system.
"""

import pyviennacl as p
import numpy as np
import random
from util import read_mtx

A = read_mtx("mat65k.mtx")
#print(A)

# We want a square N x N system.
N = 5 

# Create a random matrix with float32 precision to hold the data on the host.
#A = np.random.rand(N, N).astype(np.float32) * 10.0

raise Exception()

# Transfer the system matrix to the compute device
A = p.Matrix(A)

print("A is\n%s" % A)

# Create a right-hand-side vector on the host with random elements
# and transfer it to the compute device
b = p.Vector(np.random.rand(N).astype(np.float32) * 10.0)

print("b is %s" % b)

# Construct the tag to denote the GMRES solver
tag = p.gmres_tag(tolerance = 1e-8, max_iterations = 300, krylov_dim = 20)

# Solve the system
x = p.solve(A, b, tag)
            
# Copy the solution from the device to host and display it
print("Solution of Ax = b for x:\n%s" % x)

print(tag.vcl_tag, tag.vcl_tag.error, tag.vcl_tag.iters, tag.vcl_tag.krylov_dim,
      tag.vcl_tag.max_iterations, tag.vcl_tag.max_restarts, tag.vcl_tag.tolerance)

