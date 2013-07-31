from . import _viennacl as _v
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type,
                   int8, int16, int32, int64,
                   uint8, uint16, uint32, uint64,
                   float16, float32, float64)

# This dict is used to map NumPy dtypes onto OpenCL/ViennaCL scalar types
HostScalarTypes = {
    'int8': _v.statement_node_type.CHAR_TYPE,
    'int16': _v.statement_node_type.SHORT_TYPE,
    'int32': _v.statement_node_type.INT_TYPE,
    'int64': _v.statement_node_type.LONG_TYPE,
    'uint8': _v.statement_node_type.UCHAR_TYPE,
    'uint16': _v.statement_node_type.USHORT_TYPE,
    'uint32': _v.statement_node_type.UINT_TYPE,
    'uint64': _v.statement_node_type.ULONG_TYPE,
    'float16': _v.statement_node_type.HALF_TYPE,
    'float32': _v.statement_node_type.FLOAT_TYPE,
    'float64': _v.statement_node_type.DOUBLE_TYPE,
    'float': _v.statement_node_type.DOUBLE_TYPE
}

# This dict maps ViennaCL scalar types onto the C++ strings used for them
vcl_dtype_strings = {
    _v.statement_node_type.COMPOSITE_OPERATION_TYPE: 'index',
    _v.statement_node_type.CHAR_TYPE: 'char',
    _v.statement_node_type.UCHAR_TYPE: 'uchar',
    _v.statement_node_type.SHORT_TYPE: 'short',
    _v.statement_node_type.USHORT_TYPE: 'ushort',
    _v.statement_node_type.INT_TYPE: 'int',
    _v.statement_node_type.UINT_TYPE: 'uint',
    _v.statement_node_type.LONG_TYPE: 'long',
    _v.statement_node_type.ULONG_TYPE: 'ulong',
    _v.statement_node_type.HALF_TYPE: 'half',
    _v.statement_node_type.FLOAT_TYPE: 'float',
    _v.statement_node_type.DOUBLE_TYPE: 'double',
}

# This dict maps ViennaCL container type families onto the strings used for them
vcl_container_type_strings = {
    _v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY: 'node',
    _v.statement_node_type_family.HOST_SCALAR_TYPE_FAMILY: 'host',
    _v.statement_node_type_family.SCALAR_TYPE_FAMILY: 'scalar',
    _v.statement_node_type_family.VECTOR_TYPE_FAMILY: 'vector'
}


class NoResult: 
    """
    This no-op class is used to represent when some ViennaCL operation produces
    no explicit result, aside from any effects it may have on the operands.
    
    For instance, in-place operations can return NoResult, as can Assign.
    """
    pass


class Leaf:
    """
    This is the base class for all ``leaves`` in the ViennaCL expression tree
    system. A leaf is any type that can store data for use in an operation,
    such as a scalar, a vector, or a matrix.
    """

    def __init__(self, *args, **kwargs):
        if 'dtype' in kwargs.keys():
            dt = dtype(kwargs['dtype'])
            self.dtype = dt
        else:
            self.dtype = None
        self.statement_node_type = HostScalarTypes[self.dtype.name]
        self._init_leaf(args, kwargs)

    def _init_leaf(self, args, kwargs):
        pass

    @property
    def result_dtype(self):
        return self.dtype

    @property
    def result_container_type(self):
        return type(self)

    def express(self, statement=""):
        statement += type(self).__name__
        return statement

    def as_ndarray(self):
        return self.vcl_leaf.as_ndarray()


class Scalar(Leaf):
    """
    This is the base class for all scalar types, regardless of their memory
    and backend context. It represents the dtype and the value of the scalar
    independently of one another.

    TODO: Extend to hold ViennaCL scalar types
    + NB: Perhaps a good idea to implement vcl scalar types like numpy dtypes?
    """
    ndim = 0

    def _init_leaf(self, args, kwargs):
        if self.dtype is None:
            self.dtype = np_result_type(value)

        if 'value' in kwargs.keys():
            self.value = kwargs['value']
        else:
            self.value = 0

    @property
    def shape(self):
        raise TypeError("Scalars are 0-dimensional and thus have no shape")

    def as_ndarray(self):
        return array([self.value], dtype=self.dtype)


class HostScalar(Scalar):
    """
    This class is used to represent a ``host scalar``: a scalar type that is
    stored in main CPU RAM, and that is usually represented using a standard
    NumPy scalar dtype, such as int32 or float64.
    """
    statement_node_type_family = _v.statement_node_type_family.HOST_SCALAR_TYPE_FAMILY


class Vector(Leaf):
    ndim = 1
    statement_node_type_family = _v.statement_node_type_family.VECTOR_TYPE_FAMILY

    def _init_leaf(self, args, kwargs):
        """
        Initialise the vcl_leaf..
        """
        if self.dtype is None:
            self.dtype = float64
            vcl_type = _v.vector_double
        elif self.dtype.name == "int8":
            vcl_type = _v.vector_char
        elif self.dtype.name == "int16":
            vcl_type = _v.vector_short
        elif self.dtype.name == "int32":
            vcl_type = _v.vector_int
        elif self.dtype.name == "int64":
            vcl_type = _v.vector_long
        elif self.dtype.name == "uint8":
            vcl_type = _v.vector_uchar
        elif self.dtype.name == "uint16":
            vcl_type = _v.vector_ushort
        elif self.dtype.name == "uint32":
            vcl_type = _v.vector_uint
        elif self.dtype.name == "uint64":
            vcl_type = _v.vector_ulong
        elif self.dtype.name == "float16":
            vcl_type = _v.vector_half
        elif self.dtype.name == "float32":
            vcl_type = _v.vector_float
        elif self.dtype.name == "float64":
            vcl_type = _v.vector_double
        else:
            raise TypeError("dtype %s not supported" % self.dtype)

        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 1:
                raise TypeError("Vector can only have a 1-d shape")
            self.vcl_leaf = vcl_type(kwargs['shape'][0], 0)
        else:
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
        self.shape = (self.size,)
        self.internal_size = self.vcl_leaf.internal_size

    def __add__(self, rhs):
        return Add(self, rhs)

    def __sub__(self, rhs):
        return Sub(self, rhs)


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

        def fix_operand(opand):
            try:
                if (dtype(type(opand)).name in HostScalarTypes
                    and not (isinstance(opand, Node)
                             or isinstance(opand, Leaf))):
                    return HostScalar(opand)
                else: return opand
            except: return opand
        self.operands = list(map(fix_operand, args))

        self.result_container_type = self._result_container_type
        if self.result_container_type is NoResult:
            # Try swapping the operands
            self.operands.reverse()
            self.result_container_type = self._result_container_type
        self.dtype = self.result_dtype

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
        vcl_operand_setter = [
            "set_operand_to_",
            vcl_container_type_strings[operand.statement_node_type_family],
            "_",
            vcl_dtype_strings[operand.statement_node_type] ]
        return getattr(self.vcl_node,
                       "".join(vcl_operand_setter))

    @property
    def _result_container_type(self):
        """
        """
        if isinstance(self, Statement):
            return type(self.result)
    
        # Need to fix this for scalars.. (eg, Scalar vs HostScalar..)
        if isinstance(self, Scalar):
            return Scalar

        if isinstance(self, Leaf):
            return type(self)

        try:
            if (dtype(type(self)).name in HostScalarTypes.keys()
                and not (isinstance(self, Node))):
                return Scalar
        except: pass

        try:
            if len(self.result_types) < 1:
                return None
            return self.result_types[(
                self.operands[0].result_container_type.__name__,
                self.operands[1].result_container_type.__name__ )]
        except:
            return NoResult

    @property
    def result_dtype(self):
        if isinstance(self, Leaf):
            return self.dtype

        if isinstance(self, Statement):
            return self.result.dtype

        try:
            if dtype(type(self)).name in HostScalarTypes.keys():
                return dtype(type(self))
        except:
            pass

        return np_result_type(
            self.operands[0].result_dtype,
            self.operands[1].result_dtype )

    @property
    def result_ndim(self):
        ndim = 0
        for op in self.operands:
            if isinstance(op, Node):
                nd = op.result_ndim
                if (nd > ndim):
                    ndim = nd
            elif (op.ndim > ndim):
                ndim = op.ndim
        return ndim

    @property
    def result_max_axis_size(self):
        max_size = 1
        for op in self.operands:
            if isinstance(op, Node):
                s = op.result_max_axis_size
                if (s > max_size):
                    max_size = s
            else:
                try: op.shape
                except: continue
                for s in op.shape:
                    if (s > max_size):
                        max_size = s
        return max_size

    @property
    def result_shape(self):
        ndim = self.result_ndim
        max_size = self.result_max_axis_size
        shape = []
        for n in range(ndim):
            shape.append(max_size)
        return tuple(shape)

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
    result_types = {}
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE


class Add(Node):
    """
    Derived node class for addition
    """
    result_types = {
        ('Vector', 'Vector'): Vector
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ADD_TYPE

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
        ('Vector', 'HostScalar'): Vector
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE

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
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_SUB_TYPE

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
        elif not node.result_container_type:
            node.result_container_type = node._result_container_type
            if not node.result_container_type:
                raise TypeError("Unsupported expression: %s" % (node.express()))
            node.dtype = node.result_dtype
        else:
            self.result = node.result_container_type(
                shape = node.result_shape,
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
                elif isinstance(operand, Scalar):
                    n.get_vcl_operand_setter(operand)(op_num, operand.value)
                elif isinstance(operand, Leaf):
                    n.get_vcl_operand_setter(operand)(op_num, operand.vcl_leaf)
                else:
                    try:
                        if (dtype(type(operand)).name in HostScalarTypes.keys()
                            and not (isinstance(operand, Node))):
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
