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

# This dict maps ViennaCL scalar types onto ViennaCL vector template types
vcl_vector_types = {
#    _v.statement_node_type.CHAR_TYPE: _v.vector_char,
#    _v.statement_node_type.UCHAR_TYPE: _v.vector_uchar,
#    _v.statement_node_type.SHORT_TYPE: _v.vector_short,
#    _v.statement_node_type.USHORT_TYPE: _v.vector_ushort,
#    _v.statement_node_type.INT_TYPE: _v.vector_int,
#    _v.statement_node_type.UINT_TYPE: _v.vector_uint,
#    _v.statement_node_type.LONG_TYPE: _v.vector_long,
#    _v.statement_node_type.ULONG_TYPE: _v.vector_ulong,
#    _v.statement_node_type.HALF_TYPE: _v.vector_half,
    _v.statement_node_type.FLOAT_TYPE: _v.vector_float,
    _v.statement_node_type.DOUBLE_TYPE: _v.vector_double
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
        """
        Do initialisation tasks common to all Leaf subclasses, then pass
        control onto the overridden _init_leaf function.

        At the moment, the only task done here is to attempt to set the
        dtype and statement_node_type attributes accordingly.
        """
        if 'dtype' in kwargs.keys():    
            dt = dtype(kwargs['dtype']) 
            self.dtype = dt
        else:
            self.dtype = None
        self._init_leaf(args, kwargs)

    def _init_leaf(self, args, kwargs):
        """
        By default, leaf subclasses inherit a no-op further init function.
        
        If you're deriving from Leaf, then you probably want to override this.
        """
        pass

    @property
    def result_container_type(self):
        """
        The result_container_type for a leaf is always its own type.
        """
        return type(self)

    def express(self, statement=""):
        """
        Construct a human-readable version of a ViennaCL expression tree
        statement from this leaf.
        """
        statement += type(self).__name__
        return statement

    def as_ndarray(self):
        """
        Return a NumPy ndarray containing the data within the underlying
        ViennaCL type.
        """
        return self.vcl_leaf.as_ndarray()


class Scalar(Leaf):
    """
    This is the base class for all scalar types, regardless of their memory
    and backend context. It represents the dtype and the value of the scalar
    independently of one another.

    Because scalars are leaves in the ViennaCL expression graph, this class
    derives from the Leaf base class.

    TODO: Extend to hold ViennaCL scalar types
    + NB: Perhaps a good idea to implement vcl scalar types like numpy dtypes?
    """
    ndim = 0 # Scalars are point-like, and thus 0-dimensional

    def _init_leaf(self, args, kwargs):
        """
        Do Scalar-specific initialisation tasks.
        1. Set the scalar value to the value given, or 0.
        2. If no dtype yet set, use the NumPy type promotion rules to deduce
           a dtype.
        """
        if 'value' in kwargs.keys():
            self.value = kwargs['value']
        elif len(args) == 1:
            self.value = args[0]
        else:
            self.vaue = 0

        if self.dtype is None:
            self.dtype = np_result_type(self.value)

        self.statement_node_type = HostScalarTypes[self.dtype.name]

    @property
    def shape(self):
        raise TypeError("Scalars are 0-dimensional and thus have no shape")

    def as_ndarray(self):
        """
        Return a point-like ndarray containing only the value of this Scalar,
        with the dtype set accordingly.
        """
        return array([self.value], dtype=self.dtype)


class HostScalar(Scalar):
    """
    This class is used to represent a ``host scalar``: a scalar type that is
    stored in main CPU RAM, and that is usually represented using a standard
    NumPy scalar dtype, such as int32 or float64.

    It derives from Scalar.
    """
    statement_node_type_family = _v.statement_node_type_family.HOST_SCALAR_TYPE_FAMILY


class Vector(Leaf):
    """
    A generalised Vector class: represents ViennaCL vector objects of all
    supported scalar types. Can be constructed in a number of ways:
    + from an ndarray of the correct dtype
    + from a list
    + from an integer: produces an empty Vector of that size
    + from a tuple: first element an int (for size), second for scalar value

    Also provides convenience functions for arithmetic.
    """
    ndim = 1
    statement_node_type_family = _v.statement_node_type_family.VECTOR_TYPE_FAMILY

    def _init_leaf(self, args, kwargs):
        """
        Construct the underlying ViennaCL vector object according to the 
        given arguments and types.
        """
        if self.dtype is None:
            # TODO: Fix this 
            self.dtype = dtype(float64)

        self.statement_node_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = vcl_vector_types[self.statement_node_type]
        except:
            raise TypeError("dtype %s not supported" % self.statement_node_type)

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
    This is the base class for all nodes in the ViennaCL expression tree. A
    node is any binary or unary operation, such as addition. This class
    provides logic for expression tree construction and result type deduction,
    in order that expression statements can be executed correctly.

    If you're extending ViennaCL by adding an operation and want support for
    it in Python, then you should derive from this class.
    """
    
    statement_node_type_family = _v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY
    statement_node_type = _v.statement_node_type.COMPOSITE_OPERATION_TYPE

    def __init__(self, *args):
        """
        Take the given operand(s) to an appropriate representation for this
        operation, and deduce the result_type. Construct a ViennaCL 
        statement_node object representing this information, ready to be
        inserted into an expression statement.
        """
        if len(args) == 1:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_UNARY_TYPE_FAMILY
        elif len(args) == 2:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY
        else:
            raise TypeError("Only unary or binary nodes support currently")

        def fix_operand(opand):
            """
            If opand is a scalar type, wrap it in a PyViennaCL scalar class.
            """
            if (dtype(type(opand)).name in HostScalarTypes
                and not (isinstance(opand, Node)
                         or isinstance(opand, Leaf))):
                return HostScalar(opand)
            else: return opand
        self.operands = list(map(fix_operand, args))

        if self.result_container_type is None:
            # Try swapping the operands, in case the operation supports
            # these operand types in one order but not the other; in this case
            # the mathematical intention is not ambiguous.
            self.operands.reverse()

        # At the moment, ViennaCL does not do dtype promotion, so check that
        # the operands all have the same dtype.
        if len(self.operands) > 1:
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
        """
        This function returns the correct function for setting the
        underlying ViennaCL statement_node object's operand(s) in the correct
        way: each different container type and dtype are mapped onto a
        different set_operand_to function in the underlying object, in order
        to avoid type ambiguity at the Python/C++ interface.
        """
        vcl_operand_setter = [
            "set_operand_to_",
            vcl_container_type_strings[operand.statement_node_type_family],
            "_",
            vcl_dtype_strings[operand.statement_node_type] ]
        return getattr(self.vcl_node,
                       "".join(vcl_operand_setter))

    @property
    def result_container_type(self):
        """
        Determine the container type (ie, Scalar, Vector, etc) needed to store
        the result of the operation encoded by this Node. If the operation
        has some effect (eg, in-place), but does not produce a distinct result,
        then return NoResult. If the operation is not supported for the given
        operand types, then return None.
        """
        if len(self.result_types) < 1:
            return NoResult

        if len(self.operands) > 0:
            try:
                op0_t = self.operands[0].result_container_type.__name__
            except AttributeError:
                # Not a PyViennaCL type, so we have a number of options
                # For now, just assume HostScalar
                #
                # TODO: Fix possible alternative types
                op0_t = 'HostScalar'
        else:
            raise RuntimeError("What is a 0-ary operation?")

        if len(self.operands) > 1:
            try:
                op1_t = self.operands[1].result_container_type.__name__
                return self.result_types[(op0_t, op1_t)]
            except AttributeError:
                # Not a PyViennaCL type, so we have a number of options
                # For now, just assume HostScalar
                #
                # TODO: Fix possible alternative types
                op1_t = 'HostScalar'
            except KeyError:
                # Operation not supported for given operand types
                return None
        else:
            # No more operands, so test for 1-ary result_type
            try: return self.result_types[(op0_t, )]
            except KeyError: return None            

    @property
    def dtype(self):
        """
        Determine the dtype of the scalar element(s) of the result of the
        operation encoded by this Node, according to the NumPy type promotion
        rules.
        """
        dtypes = tuple(map(lambda x: x.dtype, self.operands))
        if len(dtypes) == 1:
            return np_result_type(dtypes[0])
        if len(dtypes) == 2:
            return np_result_type(dtypes[0], dtypes[1])

    @property
    def result_ndim(self):
        """
        Determine the maximum number of dimensions required to store the
        result of any operation on the given operands.
        """
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
        """
        Determine the maximum size of any axis required to store the result of
        any operation on the given operands.
        """
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
        """
        Determine the upper-bound shape of the object needed to store the
        result of any operation on the given operands. The len of this tuple
        is the number of dimensions, with each element of the tuple
        determining the upper-bound size of the corresponding dimension.
        """
        ndim = self.result_ndim
        max_size = self.result_max_axis_size
        shape = []
        for n in range(ndim):
            shape.append(max_size)
        return tuple(shape)

    def express(self, statement=""):
        """
        Produce a human readable representation of the expression graph
        including all nodes and leaves connected to this one, which constitutes
        the root node.
        """
        statement += type(self).__name__ + "("
        for op in self.operands:
            statement = op.express(statement) + ", "
        statement = statement[:-2] + ")"
        return statement

    def as_ndarray(self):
        """
        Returns the result of computing the operation represented by this Node
        as a NumPy ndarray instance.
        """
        s = Statement(self)
        return s.execute().as_ndarray()


class Assign(Node):
    """
    Derived node class for assignment.
    
    For example: `x = y` is represented by `Assign(x, y)`.
    """
    result_types = {}
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE


class Add(Node):
    """
    Derived node class for addition. Provides convenience magic methods for
    arithemetic.
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
    Derived node class for multiplication. Provides convenience magic methods
    for arithmetic.
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
    Derived node class for subtraction. Provides convenience magic methods
    for arithmetic.
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
    """
    This class represents the ViennaCL `statement` corresponding to an
    expression graph. It employs the type deduction information to calculate
    the resultant types, and generates the appropriate underlying ViennaCL
    object.
    """

    def __init__(self, node):
        """
        Given a Node instance, return an object representing the ViennaCL
        statement of the corresponding expression graph, as connected to the
        given root node.

        If the given root node is not an instance of Assign type, then a
        temporary object is constructed to store the result of executing the
        expression, and then a new Assign instance is created, representing
        the assignation of the result of the expression to the new temporary.
        The new Assign node is then taken to be the root node of the graph,
        having transposed the rest.
        """
        if not isinstance(node, Node):
            raise RuntimeError("Statement must be initialised on a Node")

        self.statement = []  # A list to hold the flattened expression tree
        next_node = []       # Holds nodes as we travel down the tree

        # If the root node is not an Assign instance, then construct a
        # temporary to hold the result.
        if isinstance(node, Assign):
            self.result = node.operands[0]
        elif not node.result_container_type:
            raise TypeError("Unsupported expression: %s" % (node.express()))
        else:
            self.result = node.result_container_type(
                shape = node.result_shape,
                dtype = node.dtype )
            top = Assign(self.result, node)
            next_node.append(top)

        next_node.append(node)
        # Flatten the tree
        for n in next_node:
            self.statement.append(n)
            for operand in n.operands:
                if isinstance(operand, Node):
                    next_node.append(operand)

        # Contruct a ViennaCL statement object
        self.vcl_statement = _v.statement()

        # Append the nodes in the flattened statement to the ViennaCL
        # statement, doing various type checks as we go.
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
        Execute the statement -- don't do anything else -- then return the
        result (if any).
        """
        try:
            self.vcl_statement.execute()
        except:
            print("!!! EXCEPTION EXECUTING:",self.statement[0].express())
            raise
        return self.result
