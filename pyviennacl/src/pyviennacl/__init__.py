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
@property
@classmethod -- alternate constructors
__new__
super

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

class Node:
    """
    """

    def execute(self):
        pass


class UnaryNode(Node):
    pass


class BinaryNode(Node):
    """
    BinaryNode base class -- includes shared logic for construction/execution
    """

    def __init__(self, _lhs, _rhs):
        self.statement_node_type_family = _viennacl.statement_node_type_family.COMPOSITE_OPERATION_FAMILY
        self.statement_node_type = _viennacl.statement_node_type.COMPOSITE_OPERATION_TYPE
        self.operation_node_type_family = _viennacl.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY
        self.lhs = _lhs
        self.rhs = _rhs
        self._init_node()

    def _vcl_node_factory(self):
        self.vcl_node = _viennacl.statement_node(
            self.operation_node_type_family,
            self.operation_node_type,
            self.lhs.statement_node_type_family,
            self.lhs.statement_node_type,
            self.rhs.statement_node_type_family,
            self.rhs.statement_node_type)

class Add(BinaryNode):
    """
    Derived node class for addition
    """

    def _init_node(self):
        self.operation_node_type = _viennacl.operation_node_type.OPERATION_BINARY_ADD_TYPE
        self._vcl_node_factory()


class Assign(BinaryNode):
    """
    Derived node class for assignment
    """

    def _init_node(self):
        self.operation_node_type = _viennacl.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE
        self._vcl_node_factory()


class Leaf:
    """
    Leaf base class -- generic constructors/converters..
    """

    def __init__(self):
        pass

    def as_ndarray(self):
        pass


class Vector(Leaf):
    def __init__(self):
        """
        Initialise a vcl vector -- classmethods (factories) for different types
        """
        self.statement_node_type_family = _viennacl.statement_node_type_family.VECTOR_TYPE_FAMILY
        self.statement_node_type = _viennacl.VECTOR_DOUBLE_TYPE
        self.vcl_leaf = _viennacl.vector(10, 0.1)


    @classmethod
    def scalar_vector(cls):
        """
        
        """
        pass


def get_unary_node_result_type(node):
    pass


def get_binary_node_result_type(node):
    if isinstance(node, Add):
        if isinstance(Add.lhs, Vector) and isinstance(Add.rhs, Vector):
            return Vector


def get_result_type(node):
    """
    """
    if not isinstance(node, Node):
        raise RuntimeError("Only Node instances have result types!")

    if isinstance(node, UnaryNode):
        return get_unary_node_result_type(node)

    if isinstance(node, BinaryNode):
        return get_binary_node_result_type(node)


class Statement:
    def __init__(self, node):
        """
        Take an expression tree to a statement
        
        

        """
        if not isinstance(node, Node):
            raise RuntimeError("Statement must be initialised on a Node")

        self.statement = []
        next_node = []

        if not isinstance(node, Assign):
            self.result = get_result_type(node)()
            top = Assign(result, node)
            next_node.append(top)
        else:
            self.result = node.lhs

        next_node.append(node)
        
        for n in next_node:
            self.statement.append(n)
            if isinstance(n, UnaryNode):
                pass                          
            elif isinstance(n, BinaryNode):
                # then check lhs and rhs
                if isinstance(n.lhs, Node):
                    next_node.append(n.lhs)
                if isinstance(n.rhs, Node):
                    next_node.append(n.rhs)

        self.vcl_statement = _viennacl.statement()

        for n in self.statement:
            if isinstance(n, UnaryNode):
                pass
            elif isinstance(n, BinaryNode):
                if isinstance(n.lhs, Node): # n.lhs is in statement
                    n.vcl_node.set_lhs_node_index(statement.index(n.lhs))
                elif isinstance(n.lhs, Vector): # n.lhs is a Vector reference
                    n.vcl_node.set_lhs_vector_double(n.lhs.vcl_leaf)

                if isinstance(n.rhs, Node): # n.rhs is in statement
                    n.vcl_node.set_rhs_node_index(statement.index(n.rhs))
                elif isinstance(n.rhs, Vector): # n.rhs is a Vector reference
                    n.vcl_node.set_rhs_vector_double(n.rhs.vcl_leaf)
            
            self.vcl_statement.insert_at_end(n.vcl_node)


    def execute(self):
        """
        Execute the statement -- don't do anything else!
        """
        self.vcl_statement.execute()
        return self.result


def backend_finish():
    return _viennacl.backend_finish()

