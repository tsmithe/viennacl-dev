import sys, os.path
import numpy

path = os.path.abspath(os.path.dirname(__file__))

oldpath = sys.path
sys.path.append(path)

import _viennacl

sys.path = oldpath
del path, oldpath

"""
__init__
__get_result
__get_value
__add__
__getattribute__
__setattr__

Three base classes: Statement, Leaf, and Node.

Nodes provide .evaluate() functions
.evaluate constructs a Statement instance and executes it

What are the evaluate semantics? What does it return?
* If the node on which evaluate is called is an Assign node, then
  it returns the lhs (because Assign assigns rhs to lhs).
* Otherwise, construct a new Leaf to store the result, and a new Node 
  to assign the result of the tree at this Node to the new Leaf, and then
  call .execute on the new Node, returning its result.

Leaf instances are the data interface for the expression tree
* Take data in ndarray form to ViennaCL objects
* Get data in ndarray form from ViennaCL objects
* As a convenience, can be constructed from Node instances
  + this creates a new Assign node, executes it
  + type checking? Or just leave as performed by the C++?
* Provide convenience functions for arithmetic operations:
  + __add__, 

"""

def backend_finish():
    return _viennacl.backend_finish()

