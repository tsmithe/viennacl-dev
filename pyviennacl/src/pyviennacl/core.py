from . import _viennacl as _v
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type,
                   int8, int16, int32, int64,
                   uint8, uint16, uint32, uint64,
                   float16, float32, float64)


HostScalarTypes = {
    'int8': _v.statement_node_type.HOST_SCALAR_CHAR_TYPE,
    'int16': _v.statement_node_type.HOST_SCALAR_SHORT_TYPE,
    'int32': _v.statement_node_type.HOST_SCALAR_INT_TYPE,
    'int64': _v.statement_node_type.HOST_SCALAR_LONG_TYPE,
    'uint8': _v.statement_node_type.HOST_SCALAR_UCHAR_TYPE,
    'uint16': _v.statement_node_type.HOST_SCALAR_USHORT_TYPE,
    'uint32': _v.statement_node_type.HOST_SCALAR_UINT_TYPE,
    'uint64': _v.statement_node_type.HOST_SCALAR_ULONG_TYPE,
    'float16': _v.statement_node_type.HOST_SCALAR_HALF_TYPE,
    'float32': _v.statement_node_type.HOST_SCALAR_FLOAT_TYPE,
    'float64': _v.statement_node_type.HOST_SCALAR_DOUBLE_TYPE,
    'float': _v.statement_node_type.HOST_SCALAR_DOUBLE_TYPE
}


class NoResult: pass


class Scalar:
    size = 1 ## This required because of the hacky get_vector_result_size func.

    def express(self, statement=""):
        statement += type(self).__name__
        return statement


class HostScalar(Scalar):
    """
    Convenience class to hold a scalar type and value
    
    TODO: Extend to hold ViennaCL scalar types
    + NB: Perhaps a good idea to implement vcl scalar types as numpy dtypes?
    """
    statement_node_type_family = _v.statement_node_type_family.HOST_SCALAR_TYPE_FAMILY

    def __init__(self, value=0, dtype=None):

        if dtype == None:
            self.dtype = np_result_type(value)
        else:
            self.dtype = dtype

        self.statement_node_type = HostScalarTypes[self.dtype.name]
        self.value = value

    def as_ndarray(self):
        return array(self.value)


class Leaf:
    """
    Leaf base class -- generic constructors/converters..
    """

    def express(self, statement=""):
        statement += type(self).__name__
        return statement

    def as_ndarray(self):
        return self.vcl_leaf.as_ndarray()


class Vector(Leaf):
    statement_node_type_family = _v.statement_node_type_family.VECTOR_TYPE_FAMILY

    def __init__(self, *args, **kwargs):
        """
        Initialise the vcl_leaf..
        """
        if 'dtype' in kwargs.keys():
            dt = kwargs['dtype']
            self.dtype = dt
            ## Maybe want to do this with a type conversion dict?
            ## Maybe want to generalise the type information dicts?
            if dt == int8:
                self.statement_node_type = _v.statement_node_type.VECTOR_CHAR_TYPE
                vcl_type = _v.vector_char
            elif dt == int16:
                self.statement_node_type = _v.statement_node_type.VECTOR_SHORT_TYPE
                vcl_type = _v.vector_short
            elif dt == int32:
                self.statement_node_type = _v.statement_node_type.VECTOR_INT_TYPE
                vcl_type = _v.vector_int
            elif dt == int64:
                self.statement_node_type = _v.statement_node_type.VECTOR_LONG_TYPE
                vcl_type = _v.vector_long
            elif dt == uint8:
                self.statement_node_type = _v.statement_node_type.VECTOR_UCHAR_TYPE
                vcl_type = _v.vector_uchar
            elif dt == uint16:
                self.statement_node_type = _v.statement_node_type.VECTOR_USHORT_TYPE
                vcl_type = _v.vector_ushort
            elif dt == uint32:
                self.statement_node_type = _v.statement_node_type.VECTOR_UINT_TYPE
                vcl_type = _v.vector_uint
            elif dt == uint64:
                self.statement_node_type = _v.statement_node_type.VECTOR_ULONG_TYPE
                vcl_type = _v.vector_ulong
            elif dt == float16:
                self.statement_node_type = _v.statement_node_type.VECTOR_HALF_TYPE
                vcl_type = _v.vector_half
            elif dt == float32:
                self.statement_node_type = _v.statement_node_type.VECTOR_FLOAT_TYPE
                vcl_type = _v.vector_float
            elif dt == float64:
                self.statement_node_type = _v.statement_node_type.VECTOR_DOUBLE_TYPE
                vcl_type = _v.vector_double
            else:
                raise TypeError("dtype %s not supported" % dtype)
        else:
            self.dtype = float64
            self.statement_node_type = _v.statement_node_type.VECTOR_DOUBLE_TYPE
            vcl_type = _v.vector_double

        if len(args) == 0:
            self.vcl_leaf = vcl_type()
        elif len(args) == 1:
            if isinstance(args[0], Vector):
                self.vcl_leaf = vcl_type(args[0].vcl_leaf)
            else:
                self.vcl_leaf = vcl_type(args[0])
        elif len(args) == 2:
            self.vcl_leaf = vcl_type(args[0], args[1])
        else:
            raise TypeError("Vector cannot be constructed in this way")

        self.size = self.vcl_leaf.size
        self.internal_size = self.vcl_leaf.internal_size

    def __add__(self, rhs):
        return Add(self, rhs)

    def __sub__(self, rhs):
        return Sub(self, rhs)


vcl_operand_setters = {
    _v.statement_node_type.COMPOSITE_OPERATION_TYPE: "set_operand_to_node_index",

    _v.statement_node_type.HOST_SCALAR_CHAR_TYPE: "set_operand_to_host_char",
    _v.statement_node_type.HOST_SCALAR_UCHAR_TYPE: "set_operand_to_host_uchar",
    _v.statement_node_type.HOST_SCALAR_SHORT_TYPE: "set_operand_to_host_short",
    _v.statement_node_type.HOST_SCALAR_USHORT_TYPE: "set_operand_to_host_ushort",
    _v.statement_node_type.HOST_SCALAR_INT_TYPE: "set_operand_to_host_int",
    _v.statement_node_type.HOST_SCALAR_UINT_TYPE: "set_operand_to_host_uint",
    _v.statement_node_type.HOST_SCALAR_LONG_TYPE: "set_operand_to_host_long",
    _v.statement_node_type.HOST_SCALAR_ULONG_TYPE: "set_operand_to_host_ulong",
    _v.statement_node_type.HOST_SCALAR_HALF_TYPE: "set_operand_to_host_half",
    _v.statement_node_type.HOST_SCALAR_FLOAT_TYPE: "set_operand_to_host_float",
    _v.statement_node_type.HOST_SCALAR_DOUBLE_TYPE: "set_operand_to_host_double",

    _v.statement_node_type.VECTOR_CHAR_TYPE: "set_operand_to_vector_char",
    _v.statement_node_type.VECTOR_UCHAR_TYPE: "set_operand_to_vector_uchar",
    _v.statement_node_type.VECTOR_SHORT_TYPE: "set_operand_to_vector_short",
    _v.statement_node_type.VECTOR_USHORT_TYPE: "set_operand_to_vector_ushort",
    _v.statement_node_type.VECTOR_INT_TYPE: "set_operand_to_vector_int",
    _v.statement_node_type.VECTOR_UINT_TYPE: "set_operand_to_vector_uint",
    _v.statement_node_type.VECTOR_LONG_TYPE: "set_operand_to_vector_long",
    _v.statement_node_type.VECTOR_ULONG_TYPE: "set_operand_to_vector_ulong",
    _v.statement_node_type.VECTOR_HALF_TYPE: "set_operand_to_vector_half",
    _v.statement_node_type.VECTOR_FLOAT_TYPE: "set_operand_to_vector_float",
    _v.statement_node_type.VECTOR_DOUBLE_TYPE: "set_operand_to_vector_double"
}


def get_result_container_type(node):
    """
    """
    if isinstance(node, Leaf):
        return type(node)

    if isinstance(node, Statement):
        return type(node.result)
    
    # Need to fix this for scalars.. (eg, Scalar vs HostScalar..)
    if isinstance(node, Scalar):
        return Scalar

    try:
        if (dtype(node).name in HostScalarTypes.keys()
            and not (isinstance(node, Node)
                     or isinstance(node, Leaf))):
            return Scalar
    except: pass

    try:
        if node.result_types is None:
            return None
        return node.result_types[(
            get_result_container_type(node.operands[0]).__name__,
            get_result_container_type(node.operands[1]).__name__ )]
    except:
        return NoResult


def get_result_dtype(node):
    if isinstance(node, Scalar):
        return node.dtype

    if isinstance(node, Leaf):
        return node.dtype

    if isinstance(node, Statement):
        return node.result.dtype

    return np_result_type(
        get_result_dtype(node.operands[0]),
        get_result_dtype(node.operands[1]) )


def get_vector_result_size(node):
    """
    This is just a stop-gap solution until I have a better way...
    """
    size = 0
    for op in node.operands:
        if isinstance(op, Node):
            s = get_vector_result_size(op)
            if (s > size):
                size = s
        else:
            if (op.size > size):
                size = op.size
    return size


class Node:
    """
    Node base class -- includes shared logic for construction/execution
    """
    
    statement_node_type_family = _v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY
    statement_node_type = _v.statement_node_type.COMPOSITE_OPERATION_TYPE

    def __init__(self, *args):
        if len(args) == 1:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_UNARY_TYPE_FAMILY
        elif len(args) == 2:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY
        else:
            raise TypeError("Only unary or binary nodes support currently")

        self.operands = []
        for opand in args:
            try:
                if (dtype(opand).name in HostScalarTypes.keys()
                    and not (isinstance(opand, Node)
                             or isinstance(opand, Leaf))):
                    opand = HostScalar(opand)
            except: pass
            self.operands.append(opand)

        self._init_node()

    def _vcl_node_factory(self):
        # Check result type plausibility
        self.result_container_type = get_result_container_type(self)
        if self.result_container_type is NoResult:
            # Try swapping the operands
            self.operands.reverse()
            self.result_container_type = get_result_container_type(self)
        self.dtype = get_result_dtype(self)

        # At the moment, ViennaCL does not do dtype promotion
        if dtype(self.operands[0]) != dtype(self.operands[1]):
            raise TypeError("dtypes on operands do not match: %s and %s" % (dtype(self.operands[0]), dtype(self.operands[1])))

        # Set up the ViennaCL statement_node
        self.vcl_node = _v.statement_node(
            self.operands[0].statement_node_type_family,   # lhs
            self.operands[0].statement_node_type,          # lhs
            self.operation_node_type_family,               # op
            self.operation_node_type,                      # op
            self.operands[1].statement_node_type_family,   # rhs
            self.operands[1].statement_node_type)          # rhs

    def get_vcl_operand_setter(self, operand):
        return getattr(self.vcl_node,
                       vcl_operand_setters[operand.statement_node_type])

    def express(self, statement=""):
        statement += type(self).__name__ + "("
        for op in self.operands:
            statement = op.express(statement) + ", "
        statement = statement[:-2] + ")"
        return statement

    def as_ndarray(self):
        s = Statement(self)
        return s.execute().as_ndarray()


class Assign(Node):
    """
    Derived node class for assignment
    """
    result_types = None

    def _init_node(self):
        self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE
        self._vcl_node_factory()


class Add(Node):
    """
    Derived node class for addition
    """
    result_types = {
        ('Vector', 'Vector'): Vector
    }

    def _init_node(self):
        self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_ADD_TYPE
        self._vcl_node_factory()

    def __add__(self, rhs):
        return Add(self, rhs)

    def __sub__(self, rhs):
        return Sub(self, rhs)


class Mul(Node):
    """
    Derived node class for multiplication

    Here we need to do some type deduction:
    + mat * vec -> vec
    + mat * mat -> mat
    + sca * vec -> vec
    + sca * mat -> mat
    """
    result_types = {
#        ('Vector', 'Vector'): Matrix,
        ('Vector', 'Scalar'): Vector
    }

    def _init_node(self):
        self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
        self._vcl_node_factory()

    def __add__(self, rhs):
        return Add(self, rhs)

    def __sub__(self, rhs):
        return Sub(self, rhs)


class Sub(Node):
    """
    Derived node class for addition
    """
    result_types = {
        ('Vector', 'Vector'): Vector
    }

    def _init_node(self):
        self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_SUB_TYPE
        self._vcl_node_factory()

    def __add__(self, rhs):
        return Add(self, rhs)

    def __sub__(self, rhs):
        return Sub(self, rhs)


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
        elif node.result_container_type == None:
            node.result_container_type = get_result_container_type(node)
            if node.result_container_type == None:
                raise TypeError("Unsupported expression: %s" % (node.express()))
            node.dtype = get_result_dtype(node)
        else:
            self.result = node.result_container_type(
                get_vector_result_size(node), # HACKY
                dtype = node.dtype )
            top = Assign(self.result, node)
            next_node.append(top)

        next_node.append(node)
        
        for n in next_node:
            self.statement.append(n)
            for operand in n.operands:
                if isinstance(operand, Node):
                    next_node.append(operand)

        self.vcl_statement = _v.statement()

        for n in self.statement:
            op_num = 0
            for operand in n.operands:
                if isinstance(operand, Node):
                    n.get_vcl_operand_setter(operand)(
                        op_num, 
                        self.statement.index(operand))
                elif isinstance(operand, Leaf):
                    n.get_vcl_operand_setter(operand)(op_num, operand.vcl_leaf)
                elif isinstance(operand, Scalar):
                    n.get_vcl_operand_setter(operand)(op_num, operand.value)
                else:
                    try:
                        if (dtype(operand).name in HostScalarTypes.keys()
                            and not (isinstance(operand, Node)
                                     or isinstance(operand, Leaf))):
                            n.get_vcl_operand_setter(HostScalar(operand))(
                                op_num, operand)
                    except: pass
                op_num += 1
            
            self.vcl_statement.insert_at_end(n.vcl_node)

    def execute(self):
        """
        Execute the statement -- don't do anything else!
        """
        self.vcl_statement.execute()
        return self.result
