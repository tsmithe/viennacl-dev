from pyviennacl import _viennacl as _v
from pyviennacl.pycore import (Matrix, ScalarBase, Vector, Node, MagicMethods,
                               Mul)
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type)
import logging

log = logging.getLogger(__name__)

class lower_tag:
    """
    TODO: docstring
    """
    vcl_tag = _v.lower_tag()

class unit_lower_tag:
    """
    TODO: docstring
    """
    vcl_tag = _v.unit_lower_tag()

class upper_tag:
    """
    TODO: docstring
    """
    vcl_tag = _v.upper_tag()

class unit_upper_tag:
    """
    TODO: docstring
    """
    vcl_tag = _v.unit_upper_tag()

class cg_tag:
    """
    A tag for the conjugate gradient solver.

    Used for supplying solver parameters and for dispatching the solve()
    function.
    """
    def __init__(self, tolerance = 1e-8, max_iterations = 300):
        """
        Construct a cg_tag. Parameters are:
         tolerance: Relative tolerance for the residual
                    (solver quits if ||r|| < tolerance * ||r_initial|| obtains)
         max_iterations: The maximum number of iterations
        """
        self.vcl_tag = _v.cg_tag(tolerance, max_iterations)

    @property
    def tolerance(self):
        """
        Returns the relative tolerance
        """
        return self.vcl_tag.tolerance

    @property
    def max_iterations(self):
        """
        Returns the maximum number of iterations
        """
        return self.vcl_tag.max_iterations

    @property
    def iters(self):
        """
        Returns the number of solver iterations
        """
        return self.vcl_tag.iters

    @property
    def error(self):
        """
        Returns the estimated relative error at the end of the solver run
        """
        return self.vcl_tag.error


class bicgstab_tag:
    """
    A tag for the stabilised bi-conjugate gradient (BiCGStab) solver.

    Used for supplying solver parameters and for dispatching the solve()
    function.
    """
    def __init__(self, tolerance = 1e-8, 
                 max_iterations = 400, max_iterations_before_restart = 200):
        """
        Construct a bicgstab_tag. Parameters are:
         tolerance: Relative tolerance for the residual
                    (solver quits if ||r|| < tolerance * ||r_initial|| obtains)
         max_iterations: Maximum number of iterations
         max_iterations_before restart: Maximum number of iterations before
                                        BiCGStab is reinitialised, to avoid
                                        accumulation of round-off errors.
        """
        self.vcl_tag = _v.bicgstab_tag(tolerance, max_iterations,
                                       max_iterations_before_restart)

    @property
    def tolerance(self):
        """
        Returns the relative tolerance
        """
        return self.vcl_tag.tolerance

    @property
    def max_iterations(self):
        """
        Returns the maximum number of iterations
        """
        return self.vcl_tag.max_iterations

    @property
    def max_iterations(self):
        """
        Returns the maximum number of iterations before a restart
        """
        return self.vcl_tag.max_iterations_before_restart

    @property
    def iters(self):
        """
        Returns the number of solver iterations
        """
        return self.vcl_tag.iters

    @property
    def error(self):
        """
        Returns the estimated relative error at the end of the solver run
        """
        return self.vcl_tag.error


class gmres_tag:
    """
    A tag for the GMRES solver.

    Used for supplying solver parameters and for dispatching the solve()
    function.
    """
    def __init__(self,tolerance = 1e-8, max_iterations = 300, krylov_dim = 20):
        """
        Construct a gmres_tag. Parameters are:
         tolerance: Relative tolerance for the residual
                    (solver quits if ||r|| < tolerance * ||r_initial|| obtains)
         max_iterations: Maximum number o iterations, including restarts
         krylov_dim: The maximum dimension of the Krylov space before restart
                     (number of restarts found then by
                         max_iterations / krylov_dim )
        """
        self.vcl_tag = _v.gmres_tag(tolerance, max_iterations, krylov_dim)

    @property
    def tolerance(self):
        """
        Returns the relative tolerance
        """
        return self.vcl_tag.tolerance

    @property
    def max_iterations(self):
        """
        Returns the maximum number of iterations
        """
        return self.vcl_tag.max_iterations

    @property
    def krylov_dim(self):
        """
        Returns the maximum dimension of the Krylov space before restart
        """
        return self.vcl_tag.krylov_dim

    @property
    def max_restarts(self):
        """
        Returns the maximum number of GMRES restarts
        """
        return self.vcl_tag.max_restarts

    @property
    def iters(self):
        """
        Returns the number of solver iterations
        """
        return self.vcl_tag.iters

    @property
    def error(self):
        """
        Returns the estimated relative error at the end of the solver run
        """
        return self.vcl_tag.error


class power_iter_tag:
    """
    A tag for the power iteration eigenvalue algorithm.

    Used for supplying parameters and for dispatching the eig() function.
    """
    def __init__(self, factor = 1e-8, max_iterations = 50000):
        """
        Construct a power_iter_tag. Parameters are:
         factor: Halt when the eigenvalue does not change more than this value.
         max_iterations: Maximum number of iterations to compute.
        """
        self.vcl_tag = _v.power_iter_tag(factor, max_iterations)

    @property
    def factor(self):
        """
        Returns the termination factor.

        If the eigenvalue does not change more than this value, the algorithm
        stops.
        """
        return self.vcl_tag.factor

    @property
    def max_iterations(self):
        """
        Returns the maximum number of iterations
        """
        return self.vcl_tag.max_iterations


class lanczos_tag:
    """
    A tag for the Lanczos eigenvalue algorithm.

    Used for supplying parameters and for dispatching the eig() function.
    """
    def __init__(self, factor = 0.75, num_eig = 10, method = 0, krylov = 100):
        """
        Construct a lanczos_tag. Parameters are:
         factor: Exponent of epsilon (reorthogonalisation batch tolerance)
         num_eig: Number of eigenvalues to return
         method: 0 for partial reorthogonalisation
                 1 for full reorthogonalisation
                 2 for Lanczos without reorthogonalisation
         krylov: Maximum Krylov-space size.
        """
        self.vcl_tag = _v.lanczos_tag(factor, num_eig, method, krylov)

    @property
    def factor(self):
        """
        Returns the tolerance factor for reorthogonalisation batches,
        expressed as the exponent of epsilon.
        """
        return self.vcl_tag.factor

    @property
    def num_eigenvalues(self):
        """
        Returns the number of eigenvalues to return.
        """
        return self.vcl_tag.num_eigenvalues

    @property
    def krylov_size(self):
        """
        Returns the size of the Kylov space.
        """
        return self.vcl_tag.krylov_size

    @property
    def method(self):
        """
        Returns the reorthogonalisation method choice.
        """
        return self.vcl_tag.method


def plane_rotation(vec1, vec2, alpha, beta):
    """
    TODO
    """
    # Do an assortment of type and dtype checks...
    if isinstance(vec1, Node):
        vec1 = vec1.result
    if isinstance(vec2, Node):
        vec2 = vec2.result
    if isinstance(alpha, Node):
        alpha = alpha.result
    if isinstance(beta, Node):
        beta = beta.result
    if isinstance(vec1, Vector):
        x = vec1.vcl_leaf
        if isinstance(vec2, Vector):
            if vec1.dtype != vec2.dtype:
                raise TypeError("Vector dtypes must be the same")
            y = vec2.vcl_leaf
        else:
            y = vec2
    else:
        x = vec1
        if isinstance(vec2, Vector):
            y = vec2.vcl_leaf
        else:
            y = vec2

    if isinstance(alpha, ScalarBase):
        if isinstance(vec1, Vector):
            if alpha.dtype != vec1.dtype:
                raise TypeError("Vector and scalar dtypes must be the same")
        a = alpha.value
    else:
        a = alpha

    if isinstance(beta, ScalarBase):
        if isinstance(vec1, Vector):
            if beta.dtype != vec1.dtype:
                raise TypeError("Vector and scalar dtypes must be the same")
        b = beta.value
    else:
        b = beta

    return _v.plane_rotation(x, y, a, b)


def norm(x, ord=None):
    return x.norm(ord)


def prod(A, B):
    if not isinstance(A, MagicMethods):
        return Mul(A, B)
    return (A * B)


def solve(A, B, tag):
    """
    TODO: docstring

    A must be a Matrix
    B can be Matrix or Vector
    tag can be upper_tag, lower_tag, unit_upper_tag, unit_lower_tag,
     cg_tag, bicgstab_tag or gmres_tag -- configured if necessary.
    """
    if not isinstance(A, Matrix):
        raise TypeError("A must be Matrix type")

    if isinstance(B, Matrix):
        result_type = Matrix
    elif isinstance(B, Vector):
        result_type = Vector
    else:
        raise TypeError("B must be Matrix or Vector type")

    try:
        return result_type(A.vcl_leaf.solve(B.vcl_leaf, tag.vcl_tag),
                           dtype = B.dtype,
                           layout = B.layout)
    except AttributeError:
        raise TypeError("tag must be a supported solver tag!")
Matrix.solve = solve # for convenience..


def eig(A, tag):
    """
    TODO: docstring

    A must be a Matrix
    tag can be either power_iter_tag(), or lanczos_tag(),
     configured if necessary.

    Return type depends on tag.
     - if power_iter, then a scalar of type dtype(A)
     - if lanczos, then an ndarray vector with same dtype as A
    """
    if not isinstance(A, Matrix):
        raise TypeError("A must be a Matrix type")

    if isinstance(tag, power_iter_tag):
        return _v.eig(A.vcl_leaf, tag.vcl_tag)
    elif isinstance(tag, lanczos_tag):
        return _v.eig(A.vcl_leaf, tag.vcl_tag).as_ndarray()
    else:
        raise TypeError("tag must be a supported eigenvalue tag!")
Matrix.eig = eig

def ilu(A, config):
    return NotImplemented


## And QR decomposition..?
