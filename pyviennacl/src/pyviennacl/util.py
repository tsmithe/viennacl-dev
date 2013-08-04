from . import _viennacl
import logging

default_log_handler = logging.StreamHandler()
default_log_handler.setFormatter(logging.Formatter(
    "%(levelname)s %(asctime)s %(name)s %(lineno)d %(funcName)s\n  %(message)s"
))
logging.getLogger('pyviennacl').addHandler(default_log_handler)

def backend_finish():
    return _viennacl.backend_finish()

