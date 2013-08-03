from pyviennacl import _viennacl as _v
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type)

lower_tag = _v.lower_tag
unit_lower_tag = _v.unit_lower_tag
upper_tag = _v.upper_tag
unit_upper_tag = _v.unit_upper_tag

def solve(A, B, tag):
    pass

def eig(A, tag):
    pass

def ilu(A, config):
    pass

## And QR decomposition..?
