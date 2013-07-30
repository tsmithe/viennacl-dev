from . import _viennacl

def backend_finish():
    return _viennacl.backend_finish()

