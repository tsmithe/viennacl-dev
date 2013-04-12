import sys, os.path

path = os.path.abspath(os.path.dirname(__file__))

oldpath = sys.path
sys.path.append(path)

from _viennacl import *

sys.path = oldpath

del path, oldpath
