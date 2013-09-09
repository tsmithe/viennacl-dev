from . import _viennacl
import logging

default_log_handler = logging.StreamHandler()
default_log_handler.setFormatter(logging.Formatter(
    "%(levelname)s %(asctime)s %(name)s %(lineno)d %(funcName)s\n  %(message)s"
))
logging.getLogger('pyviennacl').addHandler(default_log_handler)

def backend_finish():
    return _viennacl.backend_finish()

def from_ndarray(obj):
    if obj.ndim == 1:
        new = Vector(obj)
    elif obj.ndim == 2:
        new = Matrix(obj)
    else:
        raise AttributeError("Cannot cope with %d dimensions!" % self.operands[0].ndim)
    return new

