"""
This submodule provides convenience functions for performing basic mathematical
operations, akin to the standard Python module `math'.
"""

from pyviennacl import pycore

abs = pycore.ElementAbs
acos = pycore.ElementAcos
asin = pycore.ElementAsin
atan = pycore.ElementAtan
ceil = pycore.ElementCeil
cos = pycore.ElementCos
cosh = pycore.ElementCosh
exp = pycore.ElementExp
fabs = pycore.ElementFabs
floor = pycore.ElementFloor
log = pycore.ElementLog
log10 = pycore.ElementLog10
sin = pycore.ElementSin
sinh = pycore.ElementSinh
sqrt = pycore.ElementSqrt
tan = pycore.ElementTan
tanh = pycore.ElementTanh

