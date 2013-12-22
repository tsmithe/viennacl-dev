"""
           ext_modules=[Extension( "core",                ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/core.cpp"]),
                        Extension( "vector_int",          ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/vector_int.cpp"]),
                        Extension( "vector_long",         ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/vector_long.cpp"]),
                        Extension( "vector_uint",         ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/vector_uint.cpp"]),
                        Extension( "vector_ulong",        ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/vector_ulong.cpp"]),
                        Extension( "vector_float",        ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/vector_float.cpp"]),
                        Extension( "vector_double",       ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/vector_double.cpp"]),
                        Extension( "compressed_matrix",   ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/compressed_matrix.cpp"]),
                        Extension( "coordinate_matrix",   ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/coordinate_matrix.cpp"]),
                        Extension( "ell_matrix",          ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/ell_matrix.cpp"]),
                        Extension( "hyb_matrix",          ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/hyb_matrix.cpp"]),
                        Extension( "dense_matrix_int",    ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/dense_matrix_int.cpp"]),
                        Extension( "dense_matrix_long",   ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/dense_matrix_long.cpp"]),
                        Extension( "dense_matrix_uint",   ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/dense_matrix_uint.cpp"]),
                        Extension( "dense_matrix_ulong",  ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/dense_matrix_ulong.cpp"]),
                        Extension( "dense_matrix_float",  ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/dense_matrix_float.cpp"]),
                        Extension( "dense_matrix_double", ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/dense_matrix_double.cpp"]),
                        Extension( "eig",                 ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/eig.cpp"]),
                        Extension( "extra_functions",     ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/extra_functions.cpp"]),
                        Extension( "scheduler",           ["${CMAKE_CURRENT_SOURCE_DIR}/src/_viennacl/scheduler.cpp"])],
"""

__version__ = "1.5.0" # TODO: INTEGRATE INTO BUILD PROCESS

from .core import *

from .vector_int import *
from .vector_long import *
from .vector_uint import *
from .vector_ulong import *
from .vector_float import *
from .vector_double import *

from .compressed_matrix import *
from .coordinate_matrix import *
from .ell_matrix import *
from .hyb_matrix import *

from .dense_matrix_int import *
from .dense_matrix_long import *
from .dense_matrix_uint import *
from .dense_matrix_ulong import *
from .dense_matrix_float import *
from .dense_matrix_double import *

from .eig import *

from .extra_functions import *

from .scheduler import *
