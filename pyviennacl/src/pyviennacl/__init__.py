"""
PyViennaCL
==========

This extension provides the Python bindings for the ViennaCL linear
algebra and numerical computation library for GPGPU and heterogeneous
systems. ViennaCL itself is a header-only C++ library, so these bindings
make available to Python programmers ViennaCL's fast OpenCL and CUDA 
algorithms, in a way that is idiomatic and compatible with the Python
community's most popular scientific packages, NumPy and SciPy.

PyViennaCL is divided into four submodules, of which three are designed
for direct use by users:

  * _viennacl: a raw C++ interface to ViennaCL, with no stable API;
  * :doc:`pycore`: user-friendly classes for representing the main ViennaCL
    objects, such as Matrix or Vector;
  * :doc:`linalg`: an explicit interface to a number of ViennaCL's linear
    algebra routines, such as matrix solvers and eigenvalue computation;
  * :doc:`util`: utility functions, such as to construct an appropriate
    ViennaCL object from an ndarray (Matrix or Vector), or to provide basic
    debug logging.

Nonetheless, all of PyViennaCL's functionality is available from the top-
level pyviennacl namespace. So, if you want help on the Matrix class, you
can just run::

  >>> import pyviennacl as p
  >>> help(p.Matrix)                                     # doctest: +SKIP

However, if you want help on PyViennaCL's core functionality in general, or
PyViennaCL's high-level linear algebra functions, run::

   >>> help(p.pycore)                                    # doctest: +SKIP

or::

   >>> help(p.linalg)                                    # doctest: +SKIP
"""

from .pycore import *
from .linalg import *
from .util import *

# TODO: __all__, __version__
