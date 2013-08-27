from pyviennacl import (_viennacl as _v,
                        util)
from numpy import (ndarray, array, zeros,
                   inf, nan, dtype, 
                   equal as np_equal, array_equal,
                   result_type as np_result_type,
                   int8, int16, int32, int64,
                   uint8, uint16, uint32, uint64,
                   float16, float32, float64)
import logging, math

try:
    from scipy import sparse
    WITH_SCIPY = True
except:
    WITH_SCIPY = False

log = logging.getLogger(__name__)

# This dict maps ViennaCL container type families onto the strings used for them
#vcl_statement_node_type_family_strings = {
#    _v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY: 'node',
#    _v.statement_node_type_family.SCALAR_TYPE_FAMILY: None,
#    _v.statement_node_type_family.VECTOR_TYPE_FAMILY: None,
#    _v.statement_node_type_family.MATRIX_TYPE_FAMILY: None
#}

# This dict maps ViennaCL container subtypes onto the strings used for them
vcl_statement_node_subtype_strings = {
    _v.statement_node_subtype.INVALID_SUBTYPE: 'node',
    _v.statement_node_subtype.HOST_SCALAR_TYPE: 'host',
    _v.statement_node_subtype.DEVICE_SCALAR_TYPE: 'scalar',
    _v.statement_node_subtype.DENSE_VECTOR_TYPE: 'vector',
    _v.statement_node_subtype.IMPLICIT_VECTOR_TYPE: 'implicit_vector',
    _v.statement_node_subtype.DENSE_ROW_MATRIX_TYPE: 'matrix_row',
    _v.statement_node_subtype.DENSE_COL_MATRIX_TYPE: 'matrix_col',
    _v.statement_node_subtype.IMPLICIT_MATRIX_TYPE: 'implicit_matrix',
    _v.statement_node_subtype.COMPRESSED_MATRIX_TYPE: 'compressed_matrix',
    _v.statement_node_subtype.COORDINATE_MATRIX_TYPE: 'coordinate_matrix',
    _v.statement_node_subtype.ELL_MATRIX_TYPE: 'ell_matrix',
    _v.statement_node_subtype.HYB_MATRIX_TYPE: 'hyb_matrix'
}

# This dict maps ViennaCL numeric types onto the C++ strings used for them
vcl_statement_node_numeric_type_strings = {
    _v.statement_node_numeric_type.INVALID_NUMERIC_TYPE: 'index',
    _v.statement_node_numeric_type.CHAR_TYPE: 'char',
    _v.statement_node_numeric_type.UCHAR_TYPE: 'uchar',
    _v.statement_node_numeric_type.SHORT_TYPE: 'short',
    _v.statement_node_numeric_type.USHORT_TYPE: 'ushort',
    _v.statement_node_numeric_type.INT_TYPE: 'int',
    _v.statement_node_numeric_type.UINT_TYPE: 'uint',
    _v.statement_node_numeric_type.LONG_TYPE: 'long',
    _v.statement_node_numeric_type.ULONG_TYPE: 'ulong',
    _v.statement_node_numeric_type.HALF_TYPE: 'half',
    _v.statement_node_numeric_type.FLOAT_TYPE: 'float',
    _v.statement_node_numeric_type.DOUBLE_TYPE: 'double',
}

# This dict is used to map NumPy dtypes onto OpenCL/ViennaCL numeric types
HostScalarTypes = {
    'int8': _v.statement_node_numeric_type.CHAR_TYPE,
    'int16': _v.statement_node_numeric_type.SHORT_TYPE,
    'int32': _v.statement_node_numeric_type.INT_TYPE,
    'int64': _v.statement_node_numeric_type.LONG_TYPE,
    'uint8': _v.statement_node_numeric_type.UCHAR_TYPE,
    'uint16': _v.statement_node_numeric_type.USHORT_TYPE,
    'uint32': _v.statement_node_numeric_type.UINT_TYPE,
    'uint64': _v.statement_node_numeric_type.ULONG_TYPE,
    'float16': _v.statement_node_numeric_type.HALF_TYPE,
    'float32': _v.statement_node_numeric_type.FLOAT_TYPE,
    'float64': _v.statement_node_numeric_type.DOUBLE_TYPE,
    'float': _v.statement_node_numeric_type.DOUBLE_TYPE
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


class NoResult(object): 
    """
    This no-op class is used to represent when some ViennaCL operation produces
    no explicit result, aside from any effects it may have on the operands.
    
    For instance, in-place operations can return NoResult, as can Assign.
    """
    pass


class MagicMethods(object):
    """
    A class to provide convenience methods for arithmetic and BLAS access.

    Classes derived from this will inherit lots of useful goodies.
    """
    flushed = False

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
        return ElementProd(self, rhs)
    element_mul = element_prod

    def element_div(self, rhs):
        return ElementDiv(self, rhs)

    def __eq__(self, rhs):
        if self.flushed:
            if isinstance(rhs, MagicMethods):
                return np_equal(self.as_ndarray(), rhs.as_ndarray())
            elif isinstance(rhs, ndarray):
                return np_equal(self.as_ndarray(), rhs)
            else:
                return self.value == rhs
        else:
            return self.result == rhs

    def __hash__(self):
        return object.__hash__(self)

    def __contains__(self, item):
        return (item in self.as_ndarray())

    def __str__(self):
        return self.value.__str__()

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
        if isinstance(self, Node):
            return Add(self, rhs)
        else:
            op = InplaceAdd(self, rhs)
            op.execute()
            return self

    def __isub__(self, rhs):
        if isinstance(self, Node):
            return Sub(self, rhs)
        else:
            op = InplaceSub(self, rhs)
            op.execute()
            return self

    def __radd__(self, rhs):
        return self + rhs

    def __rsub__(self, rhs):
        return self - rhs

    def __rmul__(self, rhs):
        return self * rhs

    def __rtruediv__(self, rhs):
        return self // rhs


class View(object):
    start = None
    stop = None
    step = None

    # TODO: DOCSTRINGS
    def __init__(self, key, axis_size):
        start, stop, step = key.indices(axis_size)

        if step == 1:
            # then range -- or slice!
            self.vcl_view = _v.slice(start, 1, (stop-start))
        else:
            # then slice
            self.vcl_view = _v.slice(start, step,
                                     int(math.ceil((stop-start)/step)))

        self.slice = key
        self.start = start
        self.stop = stop
        self.step = step


class Leaf(MagicMethods):
    """
    This is the base class for all ``leaves`` in the ViennaCL expression tree
    system. A leaf is any type that can store data for use in an operation,
    such as a scalar, a vector, or a matrix.
    """
    shape = None # No shape yet -- not even 0 dimensions
    flushed = True # Are host and device data synchronised?

    def __init__(self, *args, **kwargs):
        """
        Do initialisation tasks common to all Leaf subclasses, then pass
        control onto the overridden _init_leaf function.

        TODO: UPDATE THIS

        At the moment, the only task done here is to attempt to set the
        dtype and statement_node_type attributes accordingly.
        """
        if 'dtype' in kwargs.keys():    
            dt = dtype(kwargs['dtype']) 
            self.dtype = dt
        else:
            self.dtype = None
            
        if 'view_of' in kwargs.keys():
            self.view_of = kwargs['view_of']
        if 'view' in kwargs.keys():
            self.view = kwargs['view']

        self._init_leaf(args, kwargs)

    def __setitem__(self, key, value):
        if isinstance(value, Node):
            value = value.result
        if type(self[key]) != type(value):
            raise TypeError("Cannot assign %s to %s" % (type(value),
                                                        type(self[key])))
        if self[key].dtype != value.dtype:
            raise TypeError("Cannot assign across different dtypes!")
        if self[key].shape != value.shape:
            raise TypeError("Cannot assign across different shapes!")
        Assign(self[key], value).execute()

    def _init_leaf(self, args, kwargs):
        """
        By default, leaf subclasses inherit a no-op further init function.
        
        If you're deriving from Leaf, then you probably want to override this.
        """
        raise NotImplementedError("Help")

    def flush(self):
        raise NotImplementedError("Should you be trying to flush this type?")

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
    statement_node_type_family = _v.statement_node_type_family.SCALAR_TYPE_FAMILY
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
            self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]
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
    statement_node_subtype = _v.statement_node_subtype.HOST_SCALAR_TYPE
    
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
    statement_node_subtype = _v.statement_node_subtype.DEVICE_SCALAR_TYPE

    def _init_scalar(self):
        try:
            vcl_type = getattr(_v, "scalar_" + vcl_statement_node_numeric_type_strings[self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("ViennaCL type %s not supported" % self.statement_node_numeric_type)
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
    statement_node_subtype = _v.statement_node_subtype.DENSE_VECTOR_TYPE

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
            elif isinstance(args[0], _v.vector_base):
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return args[0]
            else:
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return vcl_t(args[0])
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

        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v,
                               "vector_" + vcl_statement_node_numeric_type_strings[
                                   self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError(
                "dtype %s not supported" % self.statement_node_numeric_type)
        self.vcl_leaf = get_leaf(vcl_type)
        self.size = self.vcl_leaf.size
        self.shape = (self.size,)
        self.internal_size = self.vcl_leaf.internal_size

    def __getitem__(self, key):
        if isinstance(key, slice):
            view = View(key, self.size)
            return Vector(_v.project(self.vcl_leaf,
                                     view.vcl_view),
                          dtype=self.dtype,
                          view_of=self,
                          view=(view,))
        elif isinstance(key, tuple) or isinstance(key, list):
            if len(key) == 0:
                return self
            elif len(key) == 1:
                return self[key[0]]
            else:
                raise IndexError("Too many indices")
        else:
            # key is probably an int
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

    #@deprecated
    def __mul__(self, rhs):
        if isinstance(rhs, Vector):
            return ElementProd(self, rhs)
            #return Vector(self.vcl_leaf * rhs.vcl_leaf, dtype=self.dtype)
        else:
            op = Mul(self, rhs)
            return op


class SparseMatrixBase(Leaf):
    """
    TODO: docstring
    """
    ndim = 2
    flushed = False
    statement_node_type_family = _v.statement_node_type_family.MATRIX_TYPE_FAMILY

    @property
    def vcl_leaf_factory(self):
        raise NotImplementedError("This is only a base class!")

    def _init_leaf(self, args, kwargs):
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 2:
                raise TypeError("Sparse matrix can only have a 2-d shape")
            args = list(args)
            args.insert(0, kwargs['shape'])

        if 'layout' in kwargs.keys():
            if kwargs['layout'] == COL_MAJOR:
                raise TypeError("COL_MAJOR sparse layout not yet supported")
                self.layout = COL_MAJOR
            else:
                self.layout = ROW_MAJOR
        else:
            self.layout = ROW_MAJOR

        if len(args) == 0:
            # 0: empty -> empty
            def get_cpu_leaf(cpu_t):
                return cpu_t()
        elif len(args) == 1:
            if isinstance(args[0], tuple):
                if len(args[0]) == 2:
                    # 1: 2-tuple -> shape
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(args[0][0], args[0][1])
                elif len(args[0]) == 3:
                    # 1: 3-tuple -> shape+nnz
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(args[0][0], args[0][1], args[0][2])
                else:
                    # error!
                    raise TypeError("Sparse matrix cannot be constructed thus")
            elif isinstance(args[0], Matrix):
                # 1: Matrix instance -> copy
                self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_cpu_leaf(cpu_t):
                    return cpu_t(args[0].as_ndarray())
            elif isinstance(args[0], SparseMatrixBase):
                # 1: SparseMatrixBase instance -> copy
                self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_cpu_leaf(cpu_t):
                    return args[0].cpu_leaf
            elif isinstance(args[0], Node):
                # 1: Node instance -> get result and copy
                result = args[0].result
                if not isinstance(result, SparseMatrixBase):
                    raise TypeError("Sparse matrix cannot be constructed thus")
                self.dtype = result.dtype
                self.layout = result.layout
                def get_cpu_leaf(cpu_t):
                    return result.cpu_leaf
            elif isinstance(args[0], ndarray):
                # 1: ndarray -> init and fill
                def get_cpu_leaf(cpu_t):
                    return cpu_t(args[0])
            else:
                if WITH_SCIPY:
                    # then test for scipy.sparse matrix
                    raise NotImplementedError("SciPy support comes later")
                else:
                    # error!
                    raise TypeError("Sparse matrix cannot be constructed thus")
        elif len(args) == 2:
            # 2: 2 ints -> shape
            def get_cpu_leaf(cpu_t):
                return cpu_t(args[0], args[1])
        elif len(args) == 3:
            # 3: 3 ints -> shape+nnz
            def get_cpu_leaf(cpu_t):
                return cpu_t(args[0], args[1], args[2])
        else:
            raise TypeError("Sparse matrix cannot be constructed thus")

        if self.dtype is None:
            self.dtype = dtype(float64)            
        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            self.cpu_leaf_type = getattr(
                _v,
                "cpu_compressed_matrix_" + 
                vcl_statement_node_numeric_type_strings[
                    self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_numeric_type)

        self.cpu_leaf = get_cpu_leaf(self.cpu_leaf_type)
        self.base = self

    @property
    def nonzeros(self):
        return self.cpu_leaf.nonzeros

    @property
    def nnz(self):
        return self.cpu_leaf.nnz

    @property
    def size1(self):
        return self.cpu_leaf.size1

    @property
    def size2(self):
        return self.cpu_leaf.size2

    @property
    def size(self):
        return self.size1 * self.size2 # Flat size

    @property
    def shape(self):
        return (self.size1, self.size2)

    def resize(self, size1, size2):
        self.flushed = False
        return self.cpu_leaf.resize(size1, size2)

    def as_ndarray(self):
        return self.cpu_leaf.as_ndarray()

    def as_dense(self):
        return Matrix(self)

    @property
    def vcl_leaf(self):
        if not self.flushed:
            self.flush()
        return self._vcl_leaf

    def __getitem__(self, key):
        # TODO: extend beyond tuple keys
        if not isinstance(key, tuple):
            raise KeyError("Key must be a 2-tuple")
        if len(key) != 2:
            raise KeyError("Key must be a 2-tuple")
        return self.cpu_leaf.get(key[0], key[1])

    def __setitem__(self, key, value):
        """
        TODO:
        + Set a key-value pair (key as 2-tuple)
        + More key possibilities
        """
        if not isinstance(key, tuple):
            raise KeyError("Key must be a 2-tuple")
        if len(key) != 2:
            raise KeyError("Key must be a 2-tuple")
        self.flushed = False
        self.cpu_leaf.set(key[0], key[1], (self[key] + value))
        self.nnz # Updates nonzero list

    def __delitem__(self, key):
        if not isinstance(key, tuple):
            raise KeyError("Key must be a 2-tuple")
        if len(key) != 2:
            raise KeyError("Key must be a 2-tuple")
        self.flushed = False
        self[key] = 0
        self.nnz # Updates nonzero list

    def __str__(self):
        out = []
        for coord in self.nonzeros:
            out += ["(", "{}".format(coord[0]), ",", "{}".format(coord[1]),
                    ")\t\t", "{}".format(self[coord]), "\n"]
        out = out[:-1]
        return "".join(out)


class CompressedMatrix(SparseMatrixBase):
    """
    TODO: VCL compressed_matrix...
    """
    statement_node_subtype = _v.statement_node_subtype.COMPRESSED_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_compressed_matrix()
        self.flushed = True


class CoordinateMatrix(SparseMatrixBase):
    """
    TODO: VCL coordinate_matrix...
    """
    statement_node_subtype = _v.statement_node_subtype.COORDINATE_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_coordinate_matrix()
        self.flushed = True


class ELLMatrix(SparseMatrixBase):
    """
    TODO: VCL ell_matrix...
    """
    statement_node_subtype = _v.statement_node_subtype.ELL_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_ell_matrix()
        self.flushed = True


class HybridMatrix(SparseMatrixBase):
    """
    TODO: VCL hyb_matrix...
    """
    statement_node_subtype = _v.statement_node_subtype.HYB_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_hyb_matrix()
        self.flushed = True


# TODO: add ndarray flushing
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
    statement_node_type_family = _v.statement_node_type_family.MATRIX_TYPE_FAMILY

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
                self.statement_node_subtype = _v.statement_node_subtype.DENSE_COL_MATRIX_TYPE
            else:
                self.layout = ROW_MAJOR
                self.statement_node_subtype = _v.statement_node_subtype.DENSE_ROW_MATRIX_TYPE
        else:
            self.layout = ROW_MAJOR
            self.statement_node_subtype = _v.statement_node_subtype.DENSE_ROW_MATRIX_TYPE

        if len(args) == 0:
            def get_leaf(vcl_t):
                return vcl_t()
        elif len(args) == 1:
            if isinstance(args[0], SparseMatrixBase):
                if self.dtype is None:
                    self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_leaf(vcl_t):
                    return vcl_t(args[0].as_ndarray())
            elif isinstance(args[0], Matrix):
                if self.dtype is None:
                    self.dtype = args[0].dtype
                def get_leaf(vcl_t):
                    return vcl_t(args[0].vcl_leaf)
            elif isinstance(args[0], tuple) or isinstance(args[0], list):
                def get_leaf(vcl_t):
                    return vcl_t(args[0][0], args[0][1])
            elif isinstance(args[0], ndarray):
                if self.dtype is None:
                    self.dtype = args[0].dtype
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

        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v,
                               "matrix_" + 
                               vcl_layout_strings[self.layout] + "_" + 
                               vcl_statement_node_numeric_type_strings[
                                   self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_numeric_type)

        self.vcl_leaf = get_leaf(vcl_type)
        self.size1 = self.vcl_leaf.size1
        self.size2 = self.vcl_leaf.size2
        self.size = self.size1 * self.size2 # Flat size
        self.shape = (self.size1, self.size2)
        self.internal_size1 = self.vcl_leaf.internal_size1
        self.internal_size2 = self.vcl_leaf.internal_size2

    def __getitem__(self, key):
        if isinstance(key, tuple) or isinstance(key, list):
            if len(key) == 0:
                return self
            elif len(key) == 1: 
                return self[key[0]]
            elif len(key) == 2:
                if isinstance(key[0], int):
                    # Choose from row
                    if isinstance(key[1], int):
                        #  (int, int) -> scalar
                        return HostScalar(self.as_ndarray()[key],
                                          dtype=self.dtype)
                    elif isinstance(key[1], slice):
                        #  (int, slice) - range/slice from row -> row vector
                        view1 = View(slice(0, key[0]), self.size1)
                        view2 = View(key[1], self.size2)
                        return Matrix(_v.project(self.vcl_leaf,
                                                 view1.vcl_view,
                                                 view2.vcl_view),
                                      dtype=self.dtype,
                                      layout=self.layout,
                                      view_of=self,
                                      view=(view1, view2))
                    else:
                        raise TypeError("Did not understand key[1]")
                elif isinstance(key[0], slice):
                    # slice of rows
                    if isinstance(key[1], int):
                        #  (slice, int) - range/slice from col -> col vector
                        view1 = View(key[0], self.size1)
                        view2 = View(slice(0, key[1]), self.size2)
                        return Matrix(_v.project(self.vcl_leaf,
                                                 view1.vcl_view,
                                                 view2.vcl_view),
                                      dtype=self.dtype,
                                      layout=self.layout,
                                      view_of=self,
                                      view=(view1, view2))
                    elif isinstance(key[1], slice):
                        #  (slice, slice) - sub-matrix
                        view1 = View(key[0], self.size1)
                        view2 = View(key[1], self.size2)
                        return Matrix(_v.project(self.vcl_leaf,
                                                 view1.vcl_view,
                                                 view2.vcl_view),
                                      dtype=self.dtype,
                                      layout=self.layout,
                                      view_of=self,
                                      view=(view1, view2))
                    else:
                        raise TypeError("Did not understand key[1]")
                else:
                    raise TypeError("Did not understand key[0]")
        elif isinstance(key, slice):
            view1 = View(key, self.size1)
            view2 = View(slice(0, 1, self.size2), self.size2)
            return Matrix(_v.project(self.vcl_leaf,
                                     view1.vcl_view,
                                     view2.vcl_view),
                          dtype=self.dtype,
                          layout=self.layout,
                          view_of=self,
                          view=(view1, view2))
        elif isinstance(key, int):
            return self[slice(key)]
        else:
            raise IndexError("Did not understand key")

    def clear(self):
        return self.vcl_leaf.clear

    @property
    def T(self):
        return Trans(self)
    trans = T


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
    statement_node_subtype = _v.statement_node_subtype.INVALID_SUBTYPE
    statement_node_numeric_type = _v.statement_node_numeric_type.INVALID_NUMERIC_TYPE

    # TODO: Construct Node from ndarray
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

        self._node_init()

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
                self.operands[0].statement_node_subtype,       # lhs
                self.operands[0].statement_node_numeric_type,  # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                self.operands[1].statement_node_type_family,   # rhs
                self.operands[1].statement_node_subtype,       # rhs
                self.operands[1].statement_node_numeric_type)  # rhs
        else:
            # Set up the ViennaCL statement_node with one operand, twice..
            self.vcl_node = _v.statement_node(
                self.operands[0].statement_node_type_family,   # lhs
                self.operands[0].statement_node_subtype,       # lhs
                self.operands[0].statement_node_numeric_type,  # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                self.operands[0].statement_node_type_family,   # rhs
                self.operands[0].statement_node_subtype,       # rhs
                self.operands[0].statement_node_numeric_type)  # rhs

        self.test_init() # Make sure we can execute

    def _node_init(self):
        pass

    def test_init(self):
        layout_test = self.layout # NB QUIRK

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
            vcl_statement_node_subtype_strings[
                operand.statement_node_subtype],
            "_",
            vcl_statement_node_numeric_type_strings[
                operand.statement_node_numeric_type] ]
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
    def layout(self):
        """
        TODO
        """
        layout = None
        if self.result_container_type == Matrix:
            for opand in self.operands:
                try:
                    next_layout = opand.layout
                except:
                    continue
                if layout is None:
                    layout = next_layout
                if (next_layout != layout) and (self.operation_node_type != _v.operation_node_type.OPERATION_BINARY_MAT_MAT_PROD_TYPE):
                    raise TypeError("Matrices do not have the same layout")
            if layout is None:
                # May as well now choose a default layout ...
                layout = p.ROW_MAJOR
        return layout

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
    def shape(self):
        """
        Determine the upper-bound shape of the object needed to store the
        result of any operation on the given operands. The len of this tuple
        is the number of dimensions, with each element of the tuple
        determining the upper-bound size of the corresponding dimension.

        TODO: IMPROVE DOCSTRING: MENTION SETTER
        """
        try:
            if isinstance(self._shape, tuple):
                return self._shape
        except: pass

        ndim = self.result_ndim
        max_size = self.result_max_axis_size
        shape = []
        for n in range(ndim):
            shape.append(max_size)
        shape = tuple(shape)
        self._shape = shape
        return shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value

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
        if not self.flushed:
            self.execute()
            return self._result
        else:
            return self._result

    def execute(self):
        """
        TODO
        """
        s = Statement(self)
        self._result = s.execute()
        self.flushed = True
        return self._result

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


class ElementAbs(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ABS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAcos(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ACOS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAsin(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ASIN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAtan(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ATAN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementCeil(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_CEIL_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementCos(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_COS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementCosh(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_COSH_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementExp(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_EXP_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementFabs(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_FABS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementFloor(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_FLOOR_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementLog(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_LOG_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementLog10(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_LOG10_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementSin(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_SIN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementSinh(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_SINH_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementSqrt(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_SQRT_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementTan(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TAN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementTanh(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TANH_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class Trans(Node):
    """
    """
    result_types = {
        ('Matrix',): Matrix,
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TRANS_TYPE

    def _node_init(self):
        self.shape = (self.operands[0].shape[1],
                             self.operands[0].shape[0])


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

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot Add two differently shaped objects!")
        self.shape = self.operands[0].shape


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

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot Add two differently shaped objects!")
        self.shape = self.operands[0].shape


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

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot Add two differently shaped objects!")
        self.shape = self.operands[0].shape


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

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot Sub two differently shaped objects!")
        self.shape = self.operands[0].shape


class Mul(Node):
    """
    Derived node class for multiplication. Provides convenience magic methods
    for arithmetic.
    """
    result_types = {
        # OPERATION_BINARY_MAT_MAT_PROD_TYPE
        ('Matrix', 'Matrix'): Matrix,
        # TODO: Sparse matrix support here

        # OPERATION_BINARY_MAT_VEC_PROD_TYPE
        ('Matrix', 'Vector'): Vector,
        ('CompressedMatrix', 'Vector'): Vector,
        ('CoordinateMatrix', 'Vector'): Vector,
        ('ELLMatrix', 'Vector'): Vector,
        ('HybridMatrix', 'Vector'): Vector,

        # "OPERATION_BINARY_VEC_VEC_PROD_TYPE" -- VEC as 1-D MAT?
        #('Vector', 'Vector'): Matrix, # TODO NOT IMPLEMENTED IN SCHEDULER

        # OPERATION_BINARY_MULT_TYPE
        #('Matrix', 'HostScalar'): Matrix,
        #('Matrix', 'Scalar'): Matrix,
        ('HostScalar', 'Matrix'): Matrix,
        ('Scalar', 'Matrix'): Matrix,
        ('Vector', 'HostScalar'): Vector,
        ('Vector', 'Scalar'): Vector,
        ('Scalar', 'Scalar'): Scalar,
        ('Scalar', 'HostScalar'): HostScalar,
        ('HostScalar', 'HostScalar'): HostScalar
        # TODO: Sparse matrix support here
    }

    def _node_init(self):
        if (self.operands[0].result_container_type == Matrix or
            issubclass(self.operands[0].result_container_type,
                       SparseMatrixBase)): # Matrix * ...
            if (self.operands[1].result_container_type == Matrix or
                issubclass(self.operands[1].result_container_type,
                           SparseMatrixBase)):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MAT_MAT_PROD_TYPE
                self.shape = (self.operands[0].shape[0],
                                     self.operands[1].shape[1])
            elif self.operands[1].result_container_type == Vector:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MAT_VEC_PROD_TYPE
                self.shape = self.operands[1].shape
            elif self.operands[1].result_container_type == Scalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            elif self.operands[1].result_container_type == HostScalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            else:
                self.operation_node_type = None
        elif self.operands[0].result_container_type == Vector: # Vector * ...
            if self.operands[1].result_container_type == Scalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            elif self.operands[1].result_container_type == HostScalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            else:
                self.operation_node_type = None
        elif self.operands[0].result_container_type == Scalar: 
            #
            # TODO
            #
            if self.operands[1].result_container_type == Matrix:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[1].shape
            else:
                self.operation_node_type = None
        elif self.operands[0].result_container_type == HostScalar:
            #
            # TODO
            #
            if self.operands[1].result_container_type == Matrix:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[1].shape
            else:
                self.operation_node_type = None
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

    # TODO: shape


class ElementProd(Node):
    """
    Derived node class for element-wise multiplication. Provides convenience
    magic methods for arithmetic.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_PROD_TYPE

    # TODO: shape


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

    # TODO: shape


class Dot(Node):
    """
    """
    result_types = {
        ('Vector', 'Vector'): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_INNER_PROD_TYPE

    # TODO: shape


class Statement:
    """
    This class represents the ViennaCL `statement` corresponding to an
    expression graph. It employs the type deduction information to calculate
    the resultant types, and generates the appropriate underlying ViennaCL
    object.
    """

    def __init__(self, root):
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
        if not isinstance(root, Node):
            raise RuntimeError("Statement must be initialised on a Node")

        self.statement = []  # A list to hold the flattened expression tree
        next_node = []       # Holds nodes as we travel down the tree

        # Test to see that we can actually do the operation
        if not root.result_container_type:
            raise TypeError("Unsupported expression: %s" %(root.express()))

        # If the root node is not an Assign instance, then construct a
        # temporary to hold the result.
        if isinstance(root, Assign):
            self.result = root.operands[0]
        else:
            self.result = root.result_container_type(
                shape = root.shape,
                dtype = root.dtype,
                layout = root.layout)
            top = Assign(self.result, root)
            next_node.append(top)

        next_node.append(root)
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
                    op_idx = 0
                    for next_op in self.statement:
                        if hash(operand) == hash(next_op):
                            break
                        op_idx += 1
                    n.get_vcl_operand_setter(operand)(op_num, op_idx)
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
