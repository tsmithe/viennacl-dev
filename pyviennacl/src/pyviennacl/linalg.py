from pyviennacl import _viennacl as _v
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type)

lower_tag = _v.lower_tag
unit_lower_tag = _v.unit_lower_tag
upper_tag = _v.upper_tag
unit_upper_tag = _v.unit_upper_tag

def norm(x, ord=None):
    return NotImplemented

def solve(A, B, tag):
    return NotImplemented

def eig(A, tag):
    return NotImplemented

def ilu(A, config):
    return NotImplemented

## And QR decomposition..?
