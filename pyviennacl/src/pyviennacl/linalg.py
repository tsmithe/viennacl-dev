from pyviennacl import _viennacl as _v
from pyviennacl import Vector, ScalarBase
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type)

lower_tag = _v.lower_tag
unit_lower_tag = _v.unit_lower_tag
upper_tag = _v.upper_tag
unit_upper_tag = _v.unit_upper_tag

def plane_rotation(vec1, vec2, alpha, beta):
    """
    """
    # Do an assortment of type and dtype checks...
    if isinstance(vec1, Vector):
        x = vec1.vcl_leaf
        if isinstance(vec2, Vector):
            if vec1.dtype != vec2.dtype:
                raise TypeError("Vector dtypes must be the same")
            y = vec2.vcl_leaf
        else:
            y = vec2
    else:
        x = vec1
        if isinstance(vec2, Vector):
            y = vec2.vcl_leaf
        else:
            y = vec2

    if isinstance(alpha, ScalarBase):
        if isinstance(vec1, Vector):
            if alpha.dtype != vec1.dtype:
                raise TypeError("Vector and scalar dtypes must be the same")
        a = alpha.value
    else:
        a = alpha

    if isinstance(beta, ScalarBase):
        if isinstance(vec1, Vector):
            if beta.dtype != vec1.dtype:
                raise TypeError("Vector and scalar dtypes must be the same")
        b = beta.value
    else:
        b = beta

    return _v.plane_rotation(x, y, a, b)

def norm(x, ord=None):
    return NotImplemented

def solve(A, B, tag):
    return NotImplemented

def eig(A, tag):
    return NotImplemented

def ilu(A, config):
    return NotImplemented

## And QR decomposition..?
