import sys, os.path
import numpy

path = os.path.abspath(os.path.dirname(__file__))

oldpath = sys.path
sys.path.append(path)

import _viennacl

sys.path = oldpath
del path, oldpath

OP_ADD = 1

class build_expression:
    pass

class expression:
    def __init__(self, lhs, rhs, op):
        # define expression as binary heap, with this instance at the top
        #
        # evaluation is delayed until a call to self.eval(...), which drops into
        #  and compiles C++ if necessary and possible
        #
        # this just sets up the heap...
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.result = None

    def eval(self, *args, **kwargs):
        """
        Currently hard-coded to perform vector addition... see plan in comments!
        """
        # construct and dispatch expression by collapsing expression heap
        #
        # expression instances lower in the heap must be evaluated first (and
        #  left to right) before this can be evaluated
        # 
        # check $HOME/.config/pyviennacl/expressions.pickle for dict converting
        #   <expression-name> <--> $HOME/.cache/pyviennacl/_<module-name>.so
        # if module exists, import and execute
        # if not, compile C++ into _<module-name>.so, save, then execute
        #
        # common functions are included in pyviennacl._viennacl, so no need for
        #  extra compilation
        #
        # runtime compilation supports dynamic back-end switching
        # however, requires C++ compiler, Boost::Python, Boost::Numpy, and, of
        #  course, viennacl...
        #
        # returns whatever type follows as the result of collapsing the
        #  expression, and also sets self.result

        if self.op == OP_ADD:
            self.result = vector(self.lhs.result._vcl_vector + self.rhs.result._vcl_vector)
                            
        return self.result

    def __get_result(self):
        if self.__dict__['result']:
            return self.__dict__['result']
        else:
            return self.eval()

    def __get_value(self):
        return self.result.value

    def __add__(self, other):
        return expression(self, other, OP_ADD)

    def __getattribute__(self, name):
        if name == "value":
            return self.__get_value()
        if name == "result":
            return self.__get_result()
        else:
            return object.__getattribute__(self, name)

class vector:
    ready = False
    size = 0
    _vcl_vector = None
    args = None

    def __init__(self, *args, **kwargs):
        """
        Prepares for delayed initialisation...

        I want to turn Python into a scientific metaprogamming language!
        """
        # support the following initialisations
        #  ... scalar_vector
        #  ... empty vector (size specified)
        #  ... empty vector (size unspecified)
        #  ... copy vector
        #  ... python list (later ndarray)
        self.args = args

    def __delayed_init(self):
        """
        Do delayed initialisation!
        """
        if len(self.args) == 0:
            pass
        elif len(self.args) == 1:
            if isinstance(self.args[0], int):
                self._set_empty_vector(self.args[0])
            elif isinstance(self.args[0], vector):
                self.__copy_vector(self.args[0])
            elif isinstance(self.args[0], _viennacl.vector):
                self.__copy_vcl_vector(self.args[0])
            elif isinstance(self.args[0], list):
                self.__set_vector_from_list(self.args[0])
            elif isinstance(self.args[0], numpy.ndarray):
                self.__set_vector_from_list(self.args[0])
        elif len(self.args) == 2:
            if (isinstance(self.args[0], int) 
                and (isinstance(self.args[1], float) 
                     or isinstance(self.args[1], int))):
                self.__set_scalar_vector(self.args[0], self.args[1])
        else:
            raise AttributeError("Too many args!")
        self.ready = True

    def __setattr__(self, name, value):
        if name == "args":
            self.__dict__['args'] = value
        elif name == "ready":
            self.__dict__['ready'] = value
        elif name == "value":
            self.__set_vector_from_list(value)
        elif name == "size":
            self.__set_size(value)
        elif name == "_vcl_vector":
            if isinstance(value, _viennacl.vector):
                self.__dict__['_vcl_vector'] = value
            else:
                raise AttributeError("_vcl_vector must be a vcl_vector!")
        else:
            raise AttributeError(name)

    def __set_scalar_vector(self, size, value):
        self._vcl_vector = _viennacl.scalar_vector(size, value)

    def __set_empty_vector(self, size = 0):
        if size < 1: pass
        self.size = size
        self._vcl_vector = _viennacl.vector(size)

    def __copy_vector(self, other):
        self._vcl_vector = other._vcl_vector # Move?

    def __copy_vcl_vector(self, other):
        self._vcl_vector = other # Or copy?

    def __set_vector_from_list(self, l):
        try:
            self._vcl_vector = _viennacl.vector_from_list(l)
        except:
            self._vcl_vector = _viennacl.vector_from_ndarray(numpy.array(l))

    def __set_size(self, n):
        if self._vcl_vector is not None:
            raise AttributeError("Vector already initialised. Construct a new one")
        else:
            self.__set_empty_vector(n)            
            self.size = n

    def __getattribute__(self, name):
        if name in ['result', 'value', 'size', '_vcl_vector']:
            if not self.ready:
                self.__delayed_init()
        if name == "result":
            return self
        if name == "value":
            return self._vcl_vector.get_value()
        return object.__getattribute__(self, name)

    def __add__(self, other):
        return expression(self, other, OP_ADD)

