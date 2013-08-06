from pyviennacl import (_viennacl as _v,
                        util)
from numpy import (ndarray, array, zeros,
                   inf, nan, dtype,
                   result_type as np_result_type,
                   int8, int16, int32, int64,
                   uint8, uint16, uint32, uint64,
                   float16, float32, float64)
import logging

log = logging.getLogger(__name__)

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
    _v.statement_node_type_family.VECTOR_TYPE_FAMILY: 'vector',
    _v.statement_node_type_family.MATRIX_ROW_TYPE_FAMILY: 'matrix_row',
    _v.statement_node_type_family.MATRIX_COL_TYPE_FAMILY: 'matrix_col'
}

# Constants for choosing matrix storage layout
ROW_MAJOR = 1
COL_MAJOR = 2

vcl_layout_strings = {
    ROW_MAJOR: 'row',
    COL_MAJOR: 'col'
}

def deprecated(func):
    """
    A decorator to make deprecation really obvious.
    """
    def deprecated_function(*args):
        log.warning("DEPRECATED FUNCTION CALL: %s" % (func))
        return func(*args)
    return deprecated_function


class NoResult: 
    """
    This no-op class is used to represent when some ViennaCL operation produces
    no explicit result, aside from any effects it may have on the operands.
    
    For instance, in-place operations can return NoResult, as can Assign.
    """
    pass


class MagicMethods:
    """
    A class to provide convenience methods for arithmetic and BLAS access.

    Classes derived from this will inherit lots of useful goodies.
    """
    def norm(self, ord=None):
        if ord == 1:
            return Norm_1(self)
        elif ord == 2:
            return Norm_2(self)
        elif ord == inf:
            return Norm_Inf(self)
        else:
            return NotImplemented

    @property
    def norm_1(self):
        return Norm_1(self).result

    @property
    def norm_2(self):
        return Norm_2(self).result

    @property
    def norm_inf(self):
        return Norm_Inf(self).result

    def prod(self, rhs):
        return (self * rhs)

    def element_prod(self, rhs):
        return ElementMul(self, rhs)
    element_mul = element_prod

    def element_div(self, rhs):
        return ElementDiv(self, rhs)

    def __add__(self, rhs):
        op = Add(self, rhs)
        return op

    def __sub__(self, rhs):
        op = Sub(self, rhs)
        return op

    def __mul__(self, rhs):
        op = Mul(self, rhs)
        return op

    def __truediv__(self, rhs):
        op = Div(self, rhs)
        return op

    def __iadd__(self, rhs):
        op = InplaceAdd(self, rhs)
        op.execute()        
        return self

    def __isub__(self, rhs):
        op = InplaceSub(self, rhs)
        op.execute()
        return self

    def __rmul__(self, rhs):
        op = Mul(self, rhs)
        return op


class Leaf(MagicMethods):
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
        
        If you're derived from Leaf, then you probably want to override this.
        """
        raise NotImplementedError("Help")

    @property
    def result_container_type(self):
        """
        The result_container_type for a leaf is always its own type.
        """
        return type(self)

    @property
    def result(self):
        return self

    def copy(self):
        return type(self)(self)

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
        return array(self.vcl_leaf.as_ndarray(), dtype=self.dtype)

    @property
    def value(self):
        return self.as_ndarray()


class ScalarBase(Leaf):
    """
    This is the base class for all scalar types, regardless of their memory
    and backend context. It represents the dtype and the value of the scalar
    independently of one another.

    Because scalars are leaves in the ViennaCL expression graph, this class
    derives from the Leaf base class.
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
            self._value = kwargs['value']
        elif len(args) > 0:
            if isinstance(args[0], ScalarBase):
                self._value = args[0].value
            else:
                self._value = args[0]
        else:
            self._value = 666

        if self.dtype is None:
            self.dtype = np_result_type(self._value)

        try:
            self.statement_node_type = HostScalarTypes[self.dtype.name]
        except KeyError:
            raise TypeError("dtype %s not supported" % self.dtype.name)
        except:
            raise

        self._init_scalar()

    def _init_scalar(self):
        raise NotImplementedError("Help!")

    @property
    def shape(self):
        raise TypeError("Scalars are 0-dimensional and thus have no shape")

    @property
    def value(self):
        return self._value

    def as_ndarray(self):
        """
        Return a point-like ndarray containing only the value of this Scalar,
        with the dtype set accordingly.
        """
        return array(self.value, dtype=self.dtype)


class HostScalar(ScalarBase):
    """
    This class is used to represent a ``host scalar``: a scalar type that is
    stored in main CPU RAM, and that is usually represented using a standard
    NumPy scalar dtype, such as int32 or float64.

    It derives from ScalarBase.
    """
    statement_node_type_family = _v.statement_node_type_family.HOST_SCALAR_TYPE_FAMILY

    def _init_scalar(self):
        self.vcl_leaf = self._value


class Scalar(ScalarBase):
    """
    This class is used to represent a ViennaCL scalar: a scalar type that is
    usually stored in OpenCL global memory, but which can be converted to a 
    HostScalar, and thence to a standard NumPy scalar dtype, such as int32 or
    float64.

    It derives from ScalarBase.
    """
    statement_node_type_family = _v.statement_node_type_family.SCALAR_TYPE_FAMILY

    def _init_scalar(self):
        try:
            vcl_type = getattr(_v, "scalar_" + vcl_dtype_strings[self.statement_node_type])
        except (KeyError, AttributeError):
            raise TypeError("ViennaCL type %s not supported" % self.statement_node_type)
        self.vcl_leaf = vcl_type(self._value)

    @property
    def value(self):
        return self.vcl_leaf.as_double()


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
        
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 1:
                raise TypeError("Vector can only have a 1-d shape")
            args = list(args)
            args.insert(0, kwargs['shape'][0])

        if len(args) == 0:
            def get_leaf(vcl_t):
                return vcl_t()
        elif len(args) == 1:
            if isinstance(args[0], Vector):
                if self.dtype is None:
                    self.dtype = args[0].dtype
                def get_leaf(vcl_t):
                    return vcl_t(args[0].vcl_leaf)
            elif isinstance(args[0], ndarray):
                self.dtype = dtype(args[0])
                def get_leaf(vcl_t):
                    return vcl_t(args[0])
            elif isinstance(args[0], int):
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return vcl_t(args[0])
            else:
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return args[0]
        elif len(args) == 2:
            if self.dtype is None:
                try:
                    self.dtype = dtype(args[1])
                except TypeError:
                    self.dtype = np_result_type(args[1])
            def get_leaf(vcl_t):
                return vcl_t(args[0], args[1])
        else:
            raise TypeError("Vector cannot be constructed in this way")

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = dtype(float64)

        self.statement_node_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v, "vector_" + vcl_dtype_strings[self.statement_node_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_type)
        self.vcl_leaf = get_leaf(vcl_type)
        self.size = self.vcl_leaf.size
        self.shape = (self.size,)
        self.internal_size = self.vcl_leaf.internal_size

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is None:
                return Vector(_v.range(self.vcl_leaf, 
                                       key.start, key.stop), 
                              dtype=self.dtype)
            else:
                return Vector(_v.slice(self.vcl_leaf,
                                       key.start, key.step, 
                                       (key.stop - key.start)),
                              dtype=self.dtype)
        else:
            # might be tuple of ints, or tuple of slices, or a combo
            # or something else..
            return self.as_ndarray()[key]

    @property
    def index_norm_inf(self):
        return self.vcl_leaf.index_norm_inf

    @deprecated
    def outer(self, rhs):
        if isinstance(rhs, Vector):
            return Matrix(self.vcl_leaf.outer(rhs.vcl_leaf),
                          dtype=self.dtype,
                          layout=COL_MAJOR) # I don't know why COL_MAJOR...
        else:
            raise TypeError("Cannot calculate the outer-product of non-vector type: %s" % type(rhs))

    def dot(self, rhs):
        return Dot(self, rhs)
    inner = dot

    def as_column(self):
        tmp = self.vcl_leaf.as_ndarray()
        tmp.resize(self.size, 1)
        return Matrix(tmp, dtype=self.dtype, layout=COL_MAJOR)

    def as_row(self):
        tmp = self.vcl_leaf.as_ndarray()
        tmp.resize(1, self.size)
        return Matrix(tmp, dtype=self.dtype, layout=ROW_MAJOR)

    def as_diag(self):
        tmp_v = self.as_ndarray()
        tmp_m = zeros((self.size, self.size), dtype=self.dtype)
        for i in range(self.size):
            tmp_m[i][i] = tmp_v[i]
        return Matrix(tmp_m, dtype=self.dtype) # Ought to update this to sparse

    @deprecated
    def __mul__(self, rhs):
        if isinstance(rhs, Vector):
            #return ElementMul(self, rhs) # TODO: ..NOT YET IMPLEMENTED
            return Vector(self.vcl_leaf * rhs.vcl_leaf, dtype=self.dtype)
        else:
            op = Mul(self, rhs)
            return op


class Matrix(Leaf):
    """
    A generalised Matrix class: represents ViennaCL matrix objects of all
    supported scalar types. Can be constructed in a number of ways:
    + from an ndarray of the correct dtype
    + from an integer tuple: produces an empty Matrix of that shape
    + from a tuple: first two values shape, third scalar value

    Also provides convenience functions for arithmetic.

    Default layout is ROW_MAJOR.

    TODO: Expand this documentation.
    """
    ndim = 2

    def _init_leaf(self, args, kwargs):
        """
        Construct the underlying ViennaCL vector object according to the 
        given arguments and types.
        """
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 2:
                raise TypeError("Matrix can only have a 2-d shape")
            args = list(args)
            args.insert(0, kwargs['shape'])

        if 'layout' in kwargs.keys():
            if kwargs['layout'] == COL_MAJOR:
                self.layout = COL_MAJOR
                self.statement_node_type_family = _v.statement_node_type_family.MATRIX_COL_TYPE_FAMILY
            else:
                self.layout = ROW_MAJOR
                self.statement_node_type_family = _v.statement_node_type_family.MATRIX_ROW_TYPE_FAMILY
        else:
            self.layout = ROW_MAJOR
            self.statement_node_type_family = _v.statement_node_type_family.MATRIX_ROW_TYPE_FAMILY

        if len(args) == 0:
            def get_leaf(vcl_t):
                return vcl_t()
        elif len(args) == 1:
            if isinstance(args[0], Matrix):
                if self.dtype is None:
                    self.dtype = args[0].dtype
                def get_leaf(vcl_t):
                    return vcl_t(args[0].vcl_leaf)
            elif isinstance(args[0], tuple) or isinstance(args[0], list):
                def get_leaf(vcl_t):
                    return vcl_t(args[0][0], args[0][1])
            elif isinstance(args[0], ndarray):
                if self.dtype is None:
                    self.dtype = dtype(args[0])
                def get_leaf(vcl_t):
                    return vcl_t(args[0])
            else:
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return args[0]
        elif len(args) == 2:
            if isinstance(args[0], tuple) or isinstance(args[0], list):
                if self.dtype is None:
                    self.dtype = np_result_type(args[1])
                def get_leaf(vcl_t):
                    return vcl_t(args[0][0], args[0][1], args[1])
            else:
                def get_leaf(vcl_t):
                    return vcl_t(args[0], args[1])
        elif len(args) == 3:
            if self.dtype is None:
                self.dtype = np_result_type(args[2])
            def get_leaf(vcl_t):
                return vcl_t(args[0], args[1], args[2])
        else:
            raise TypeError("Matrix cannot be constructed in this way")

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = dtype(float64)

        self.statement_node_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v,
                               "matrix_" + 
                               vcl_layout_strings[self.layout] + "_" + 
                               vcl_dtype_strings[self.statement_node_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_type)

        self.vcl_leaf = get_leaf(vcl_type)
        self.size1 = self.vcl_leaf.size1
        self.size2 = self.vcl_leaf.size2
        self.size = self.size1 * self.size2 # Flat size
        self.shape = (self.size1, self.size2)
        self.internal_size1 = self.vcl_leaf.internal_size1
        self.internal_size2 = self.vcl_leaf.internal_size2

    def __getitem__(self, key):
        # TODO TODO TODO TODO
        # TODO TODO TODO TODO
        # TODO TODO TODO TODO
        # TODO TODO TODO TODO
        if isinstance(key, tuple) or isinstance(key, list):
            if len(key) == 0:
                return self
            elif len(key) == 1: # Then we want a row
                # We have either (int) or (slice)
                return Matrix(_v.range(self.vcl_leaf,
                                       ROW_START, ROW_END,
                                       COL_START, COL_END),
                              dtype=self.dtype,
                              layout=self.layout)
            elif len(key) == 2:
                # Then we have one of the following:
                #  (int, int)
                #  (int, slice)
                #  (slice, int)
                #  (slice, slice)
                pass
        elif isinstance(key, slice):
            if key.step is None:
                return Matrix(_v.range(self.vcl_leaf, 
                                       key.start, key.stop), 
                              dtype=self.dtype,
                              layout=self.layout)
            else:
                return Matrix(_v.slice(self.vcl_leaf,
                                       key.start, key.step, 
                                       (key.stop - key.start)),
                              dtype=self.dtype,
                              layout=self.layout)
        elif isinstance(key, int):
            return self.as_ndarray()[key]
        else:
            raise IndexError("Did not understand key")

    def clear(self):
        return self.vcl_leaf.clear

    @deprecated
    def T(self):
        return self.vcl_leaf.trans
    trans = T

    @deprecated
    def __mul__(self, rhs):
        if isinstance(rhs, Matrix):
            return Matrix(self.vcl_leaf * rhs.vcl_leaf, dtype=self.dtype)
        elif isinstance(rhs, Vector):
            return Vector(self.vcl_leaf * rhs.vcl_leaf, dtype=self.dtype)
        else:
            op = Mul(self, rhs)
            return op


class Node(MagicMethods):
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
            if (np_result_type(opand).name in HostScalarTypes
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
            if self.result_container_type is None:
                # No use, so revert
                self.operands.reverse()

        self._extra_init()

        if self.operation_node_type is None:
            raise TypeError("Unsupported expression: %s" % (self.express()))

        # At the moment, ViennaCL does not do dtype promotion, so check that
        # the operands all have the same dtype.
        if len(self.operands) > 1:
            if dtype(self.operands[0]) != dtype(self.operands[1]):
                raise TypeError("dtypes on operands do not match: %s and %s" % (dtype(self.operands[0]), dtype(self.operands[1])))
            # Set up the ViennaCL statement_node with two operands
            self.vcl_node = _v.statement_node(
                self.operands[0].statement_node_type_family,   # lhs
                self.operands[0].statement_node_type,          # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                self.operands[1].statement_node_type_family,   # rhs
                self.operands[1].statement_node_type)          # rhs
        else:
            # Set up the ViennaCL statement_node with one operand, twice..
            self.vcl_node = _v.statement_node(
                self.operands[0].statement_node_type_family,   # lhs
                self.operands[0].statement_node_type,          # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                self.operands[0].statement_node_type_family,   # rhs
                self.operands[0].statement_node_type)          # rhs

    def _extra_init(self):
        pass

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
        Determine the container type (ie, ScalarBase, Vector, etc) needed to
        store the result of the operation encoded by this Node. If the 
        operation has some effect (eg, in-place), but does not produce a
        distinct result, then return NoResult. If the operation is not
        supported for the given operand types, then return None.
        """
        if len(self.result_types) < 1:
            return NoResult

        if len(self.operands) > 0:
            try:
                op0_t = self.operands[0].result_container_type.__name__
            except AttributeError:
                # Not a PyViennaCL type, so we have a number of options
                # However, since HostScalar can cope with any normal Pythonic
                # scalar type, we assume that and hope for the best.
                op0_t = 'HostScalar'
        else:
            raise RuntimeError("What is a 0-ary operation?")

        if len(self.operands) > 1:
            try:
                op1_t = self.operands[1].result_container_type.__name__
                return self.result_types[(op0_t, op1_t)]
            except AttributeError:
                # Not a PyViennaCL type, so we have a number of options.
                # However, since HostScalar can cope with any normal Pythonic
                # scalar type, we assume that and hope for the best.
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

    @property
    def result(self):
        """
        Returns the result of computing the operation represented by this Node.
        """
        s = Statement(self)
        return s.execute()

    def execute(self):
        """
        Synonymous with self.result
        """
        return self.result

    @property
    def value(self):
        """
        Returns the value of the result of computing the operation represented 
        by this Node.
        """
        s = Statement(self)
        return s.execute().value

    def as_ndarray(self):
        return array(self.value, dtype=self.dtype)


class Norm_1(Node):
    """
    """
    result_types = {
        ('Vector',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_1_TYPE


class Norm_2(Node):
    """
    """
    result_types = {
        ('Vector',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_2_TYPE


class Norm_Inf(Node):
    """
    """
    result_types = {
        ('Vector',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_INF_TYPE


class Assign(Node):
    """
    Derived node class for assignment.
    
    For example: `x = y` is represented by `Assign(x, y)`.
    """
    result_types = {}
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE


class InplaceAdd(Assign):
    """
    Derived node class for in-place addition. Derives from Assign rather than
    directly from Node because in-place operations are mathematically similar
    to assignation. Provides convenience magic methods for arithmetic.
    """
    result_types = {
        ('Scalar', 'Scalar'): Scalar,
        ('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type  = _v.operation_node_type.OPERATION_BINARY_INPLACE_ADD_TYPE


class InplaceSub(Assign):
    """
    Derived node class for in-place addition. Derives from Assign rather than
    directly from Node because in-place operations are mathematically similar
    to assignation. Provides convenience magic methods for arithmetic.
    """
    result_types = {
        ('Scalar', 'Scalar'): Scalar,
        ('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type  = _v.operation_node_type.OPERATION_BINARY_INPLACE_SUB_TYPE


class Add(Node):
    """
    Derived node class for addition. Provides convenience magic methods for
    arithmetic.
    """
    result_types = {
        ('Scalar', 'Scalar'): Scalar,
        ('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ADD_TYPE


class Sub(Node):
    """
    Derived node class for subtraction. Provides convenience magic methods
    for arithmetic.
    """
    result_types = {
        ('Scalar', 'Scalar'): Scalar,
        ('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_SUB_TYPE


class Mul(Node):
    """
    Derived node class for multiplication. Provides convenience magic methods
    for arithmetic.
    """
    result_types = {
        # OPERATION_BINARY_MAT_MAT_PROD_TYPE
        #('Matrix', 'Matrix'): Matrix, # NOT IMPLEMENTED IN SCHEDULER

        # OPERATION_BINARY_MAT_VEC_PROD_TYPE
        #('Matrix', 'Vector'): Matrix, # NOT IMPLEMENTED IN SCHEDULER

        # "OPERATION_BINARY_VEC_VEC_PROD_TYPE" -- VEC as 1-D MAT?
        #('Vector', 'Vector'): Matrix, # NOT IMPLEMENTED IN SCHEDULER

        # OPERATION_BINARY_MULT_TYPE
        ('Matrix', 'HostScalar'): Matrix,
        ('Matrix', 'Scalar'): Matrix,
        ('Vector', 'HostScalar'): Vector,
        ('Vector', 'Scalar'): Vector,
        #('Scalar', 'Scalar'): Scalar,
        #('Scalar', 'HostScalar'): HostScalar,
        #('HostScalar', 'HostScalar'): HostScalar
    }

    def _extra_init(self):
        if isinstance(self.operands[0], Matrix): # Matrix * ...
            if isinstance(self.operands[1], Matrix):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MAT_MAT_PROD_TYPE
            elif isinstance(self.operands[1], Vector):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MAT_VEC_PROD_TYPE
            elif isinstance(self.operands[1], Scalar):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
            elif isinstance(self.operands[1], HostScalar):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
            else:
                self.operation_node_type = None
        elif isinstance(self.operands[0], Vector): # Vector * ...
            if isinstance(self.operands[1], Scalar):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
            elif isinstance(self.operands[1], HostScalar):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
            else:
                self.operation_node_type = None
        #elif isinstance(self.operands[0], Scalar): 
        #    pass
        #elif isinstance(self.operands[0], HostScalar):
        #    pass
        else:
            self.operation_node_type = None


class Div(Node):
    """
    Derived node class for vector/matrix-scalar division. Provides convenience
    magic methods for arithmetic.
    """
    result_types = {
        ('Vector', 'Scalar'): Vector,
        ('Vector', 'HostScalar'): Vector,
        ('Matrix', 'Scalar'): Matrix,
        ('Matrix', 'HostScalar'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_DIV_TYPE


class ElementMul(Node):
    """
    Derived node class for element-wise multiplication. Provides convenience
    magic methods for arithmetic.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_MULT_TYPE


class ElementDiv(Node):
    """
    Derived node class for element-wise division. Provides convenience magic
    methods for arithmetic.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_DIV_TYPE


class Dot(Node):
    """
    """
    result_types = {
        ('Vector', 'Vector'): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_INNER_PROD_TYPE


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

        # Test to see that we can actually do the operation
        if not node.result_container_type:
            raise TypeError("Unsupported expression: %s" %(node.express()))

        # If the root node is not an Assign instance, then construct a
        # temporary to hold the result.
        if isinstance(node, Assign):
            self.result = node.operands[0]
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
                elif isinstance(operand, Leaf):
                    n.get_vcl_operand_setter(operand)(op_num, operand.vcl_leaf)
                elif np_result_type(operand).name in HostScalarTypes.keys():
                    n.get_vcl_operand_setter(HostScalar(operand))(
                        op_num, operand)
                op_num += 1
            self.vcl_statement.insert_at_end(n.vcl_node)

    def execute(self):
        """
        Execute the statement -- don't do anything else -- then return the
        result (if any).
        """
        try:
            self.vcl_statement.execute()
        except RuntimeError:
            log.error("EXCEPTION EXECUTING: %s" %(self.statement[0].express()))
            raise
        return self.result
