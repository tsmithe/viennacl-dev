import sys, os.path

path = os.path.abspath(os.path.dirname(__file__))

oldpath = sys.path
sys.path.append(path)

from _viennacl_rvalue import *

sys.path = oldpath

del path, oldpath
