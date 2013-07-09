import sys, os.path

path = os.path.abspath(os.path.dirname(__file__))

oldpath = sys.path
sys.path.append(path)

from _viennacl_numpy import *
from numpy import *

sys.path = oldpath

del path, oldpath
