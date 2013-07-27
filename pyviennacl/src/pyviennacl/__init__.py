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


It needs to be possible to register type deductions at runtime.
+ result type given n-ary node
+ vcl_node.set_rhs_*/set_lhs_* function calls

We can abstract from UnaryNode and BinaryNode to n-ary Node, which
represents not lhs and rhs, but n operands in Node.operands (a list):
Node.operands[0] = lhs; Node.operands[1] = rhs, etc
"""

vcl_operand_setters = {
    _viennacl.statement_node_type.COMPOSITE_OPERATION_TYPE: "set_operand_to_node_index",
    _viennacl.statement_node_type.VECTOR_DOUBLE_TYPE: "set_operand_to_vector_double"}


class Node:
    """
    Node base class -- includes shared logic for construction/execution
    """
    
    statement_node_type_family = _viennacl.statement_node_type_family.COMPOSITE_OPERATION_FAMILY
    statement_node_type = _viennacl.statement_node_type.COMPOSITE_OPERATION_TYPE

    def __init__(self, *args):
        if len(args) == 1:
            self.operation_node_type_family = _viennacl.operation_node_type_family.OPERATION_UNARY_TYPE_FAMILY
        elif len(args) == 2:
            self.operation_node_type_family = _viennacl.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY
        else:
            raise TypeError("Only unary or binary nodes support currently")

        self.operands = []
        for opand in args:
            #self.operands.append(Operand(self, opand))
            self.operands.append(opand)

        self._init_node()

    def _vcl_node_factory(self):
        self.vcl_node = _viennacl.statement_node(
            self.operation_node_type_family,
            self.operation_node_type,
            self.operands[0].statement_node_type_family,
            self.operands[0].statement_node_type,
            self.operands[1].statement_node_type_family,
            self.operands[1].statement_node_type)

    def get_vcl_operand_setter(self, operand):
        return getattr(self.vcl_node,
                       vcl_operand_setters[operand.statement_node_type])

    def as_ndarray(self):
        s = Statement(self)
        return s.execute().as_ndarray()


class Leaf:
    """
    Leaf base class -- generic constructors/converters..
    """

    def __init__(self):
        pass

    def as_ndarray(self):
        return self.vcl_leaf.as_ndarray()


class Vector(Leaf):
    statement_node_type_family = _viennacl.statement_node_type_family.VECTOR_TYPE_FAMILY
    statement_node_type = _viennacl.statement_node_type.VECTOR_DOUBLE_TYPE

    def __init__(self, *args):
        """
        Initialise the vcl_leaf..
        """
        if len(args) == 0:
            self.vcl_leaf = _viennacl.vector()
        elif len(args) == 1:
            if isinstance(args[0], int):
                self.vcl_leaf = _viennacl.vector(args[0])
            elif isinstance(args[0], Vector):
                self.vcl_leaf = _viennacl.vector(args[0].vcl_leaf)
            elif isinstance(args[0], numpy.ndarray):
                self.vcl_leaf = _viennacl.vector(args[0])
            elif isinstance(args[0], list):
                self.vcl_leaf = _viennacl.vector(args[0])
            else:
                raise TypeError("Vector cannot be constructed from %s instance" % type(args[0]))
        elif len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], float):
                self.vcl_leaf = _viennacl.vector(args[0], args[1])
            else:
                raise TypeError("Vector cannot be constructed from instances of %s and %s" % (type(args[0]), type(args[1])))
        else:
            raise TypeError("Vector cannot be constructed in this way")

        self.size = self.vcl_leaf.size
        self.internal_size = self.vcl_leaf.internal_size

    def __add__(self, rhs):
        return Add(self, rhs)


class Add(Node):
    """
    Derived node class for addition
    """
    result_types = {
        (_viennacl.statement_node_type.VECTOR_DOUBLE_TYPE,
         _viennacl.statement_node_type.VECTOR_DOUBLE_TYPE): Vector
    }

    def _init_node(self):
        self.operation_node_type = _viennacl.operation_node_type.OPERATION_BINARY_ADD_TYPE
        self._vcl_node_factory()

    def __add__(self, rhs):
        return Add(self, rhs)


class Assign(Node):
    """
    Derived node class for assignment
    """

    def _init_node(self):
        self.operation_node_type = _viennacl.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE
        self._vcl_node_factory()


def get_result_type(node):
    """
    """
    if isinstance(node, Leaf):
        return type(node)

    if isinstance(node, Statement):
        return type(node.result)

    return node.result_types[(
        get_result_type(node.operands[0]).statement_node_type,
        get_result_type(node.operands[1]).statement_node_type )]

    raise RuntimeError("Only Node, Leaf and Statement instances have result types!")

class Statement:
    def __init__(self, node):
        """
        Take an expression tree to a statement
        """
        if not isinstance(node, Node):
            raise RuntimeError("Statement must be initialised on a Node")

        self.statement = []
        next_node = []

        if isinstance(node, Assign):
            self.result = node.operands[0]
        else:
            self.result = get_result_type(node)(10)
            top = Assign(self.result, node)
            next_node.append(top)

        next_node.append(node)
        
        for n in next_node:
            self.statement.append(n)
            for operand in n.operands:
                if isinstance(operand, Node):
                    next_node.append(operand)

        self.vcl_statement = _viennacl.statement()

        for n in self.statement:
            op_num = 0
            for operand in n.operands:
                if isinstance(operand, Node):
                    n.get_vcl_operand_setter(operand)(
                        op_num, 
                        self.statement.index(operand))
                else:
                    n.get_vcl_operand_setter(operand)(op_num, operand.vcl_leaf)
                op_num += 1
            
            self.vcl_statement.insert_at_end(n.vcl_node)

    def execute(self):
        """
        Execute the statement -- don't do anything else!
        """
        self.vcl_statement.execute()
        return self.result


def backend_finish():
    return _viennacl.backend_finish()

