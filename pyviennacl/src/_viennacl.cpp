#include <cstdint>
#include <iostream>
#include <typeinfo>

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#define VIENNACL_WITH_UBLAS
#define VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_PYTHON
#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

namespace vcl = viennacl;
namespace bp = boost::python;
namespace np = boost::numpy;
namespace ublas = boost::numeric::ublas;

typedef void* NoneT;

typedef vcl::scalar<double> vcl_scalar_t;
typedef double              cpu_scalar_t;

typedef std::vector<cpu_scalar_t> cpu_vector_t;
typedef vcl::vector<cpu_scalar_t> vcl_vector_t;

typedef vcl::matrix<cpu_scalar_t,
		    vcl::row_major> vcl_matrix_t;
typedef ublas::matrix<cpu_scalar_t,
		  ublas::row_major> cpu_matrix_t;

typedef std::vector< std::map< uint32_t, cpu_scalar_t> > cpu_sparse_matrix_t;


typedef vcl::compressed_matrix<cpu_scalar_t> vcl_compressed_matrix_t;
/*typedef vcl::coordinate_matrix<cpu_scalar_t> vcl_coordinate_matrix_t;

typedef vcl::ell_matrix<cpu_scalar_t> vcl_ell_matrix_t;
typedef vcl::hyb_matrix<cpu_scalar_t> vcl_hyb_matrix_t;
*/

enum op_t {
  op_add,
  op_sub,
  op_mul,
  op_div,
  op_iadd,
  op_isub,
  op_imul,
  op_idiv,
  op_inner_prod,
  op_outer_prod,
  op_element_prod,
  op_element_div,
  op_norm_1,
  op_norm_2,
  op_norm_inf,
  op_index_norm_inf,
  op_plane_rotation,
  op_trans,
  op_prod,
  op_solve
};

// Generic operation dispatch class -- see specialisations below
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
struct pyvcl_worker
{
  static ReturnT do_op(void* o) {}
};

// This class wraps operations in a type-independent way up to 4 operands.
// It's mainly used to simplify and consolidate calling conventions in the 
// main module code far below, but it also includes a small amount of logic
// for the extraction of C++ types from Python objects where necessary.
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs=0>
struct pyvcl_op
{
  Operand1T operand1;
  Operand2T operand2;
  Operand3T operand3;
  Operand4T operand4;
  friend struct pyvcl_worker<ReturnT,
			     Operand1T, Operand2T,
			     Operand3T, Operand4T,
			     op, PyObjs>;
  
  pyvcl_op(Operand1T opand1, Operand2T opand2,
	   Operand3T opand3, Operand4T opand4)
    : operand1(opand1), operand2(opand2),
      operand3(opand3), operand4(opand4)
  {
    
    /*
      
      The value of the template variable PyObjs determines which operands
      need to be extracted from Python objects, by coding the operand
      "position" in binary. This is the object-extraction logic alluded to
      in the comments above.
      
      So, given (as an example) PyObjs == 7 == 0111b, and given that we 
      number operands from left to right, the following operands need
      extraction: operand2, operand3, and operand4.
      
    */
    
    if (PyObjs & 8) {
      operand1 = static_cast<Operand1T>
	(bp::extract<Operand1T>((bp::api::object)opand1));
    } else {
      operand1 = opand1;
    }
    
    if (PyObjs & 4) {
      operand2 = static_cast<Operand2T>
	(bp::extract<Operand2T>((bp::api::object)opand2));
    } else {
      operand2 = opand2;
    }
    
    if (PyObjs & 2) {
      operand3 = static_cast<Operand3T>
	(bp::extract<Operand3T>((bp::api::object)opand3));
    } else {
      operand3 = opand3;
    }
    
    if (PyObjs & 1) {
      operand4 = static_cast<Operand4T>
	(bp::extract<Operand4T>((bp::api::object)opand4));
    } else {
      operand4 = opand4;
    }
    
  }    

  // Should I just use operator(), I wonder..
  ReturnT do_op()
  {
    return pyvcl_worker<ReturnT,
			Operand1T, Operand2T,
			Operand3T, Operand4T,
			op, PyObjs>::do_op(*this);
  }
};

// Convenient operation dispatch functions.
// These functions make setting up and calling the pyvcl_op class much
// simpler for the specific 1-, 2-, 3- and 4-operand cases.

template <class ReturnT,
	  class Operand1T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_1ary_op(Operand1T a)
{
  pyvcl_op<ReturnT,
	   Operand1T, NoneT,
	   NoneT, NoneT,
	   op, PyObjs>
    o (a, NULL, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_2ary_op(Operand1T a, Operand2T b)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   NoneT, NoneT,
	   op, PyObjs>
    o (a, b, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_3ary_op(Operand1T a, Operand2T b, Operand3T c)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   Operand3T, NoneT,
	   op, PyObjs>
    o (a, b, c, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_4ary_op(Operand1T a, Operand2T b,
			 Operand3T c, Operand4T d)
  {
    pyvcl_op<ReturnT,
	     Operand1T, Operand2T,
	     Operand3T, Operand4T,
	     op, PyObjs>
      o (a, b, c, d);
  return o.do_op();
}


/*****************************
  Operation wrapper functions
 *****************************/

// These macros define specialisations of the pyvcl_worker class
// which is used to dispatch viennacl operations.

#define OP_TEMPLATE template <class ReturnT, \
                              class Operand1T, class Operand2T, \
                              class Operand3T, class Operand4T, \
                              int PyObjs>
#define PYVCL_WORKER_STRUCT(OP) OP_TEMPLATE \
                              struct pyvcl_worker<ReturnT, \
                              Operand1T, Operand2T, \
                              Operand3T, Operand4T, \
                              OP, PyObjs>
#define DO_OP_FUNC(OP) PYVCL_WORKER_STRUCT(OP) { \
                              static ReturnT do_op(pyvcl_op<ReturnT, \
                              Operand1T, Operand2T, \
                              Operand3T, Operand4T, \
                              OP, PyObjs>& o)

// And the actual operations follow below.
  
DO_OP_FUNC(op_add) { return o.operand1 + o.operand2; } };
DO_OP_FUNC(op_sub) { return o.operand1 - o.operand2; } };
DO_OP_FUNC(op_mul) { return o.operand1 * o.operand2; } };
DO_OP_FUNC(op_div) { return o.operand1 / o.operand2; } };

DO_OP_FUNC(op_inner_prod)
{
  return vcl::linalg::inner_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_outer_prod)
{
  return vcl::linalg::outer_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_element_prod)
{
  return vcl::linalg::element_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_element_div)
{
  return vcl::linalg::element_div(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_iadd)
{
  o.operand1 += o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_isub)
{
  o.operand1 -= o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_imul)
{
  o.operand1 *= o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_idiv)
{
  o.operand1 /= o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_norm_1)
{
  return vcl::linalg::norm_1(o.operand1);
} };

DO_OP_FUNC(op_norm_2)
{
  return vcl::linalg::norm_2(o.operand1);
} };

DO_OP_FUNC(op_norm_inf)
{
  return vcl::linalg::norm_inf(o.operand1);
} };

DO_OP_FUNC(op_index_norm_inf)
{
  return vcl::linalg::index_norm_inf(o.operand1);
} };

DO_OP_FUNC(op_plane_rotation)
{
  vcl::linalg::plane_rotation(o.operand1, o.operand2,
			      o.operand3, o.operand4);
  return bp::object();
} };

DO_OP_FUNC(op_trans)
{
  return vcl::trans(o.operand1);
} };

DO_OP_FUNC(op_prod)
{
  return vcl::linalg::prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_solve)
{
  return vcl::linalg::solve(o.operand1, o.operand2,
			    o.operand3);
} };

/*******************************
  Type conversion functions
 *******************************/
  
// Scalar

/** @brief Returns a double describing the VCL_T */
cpu_scalar_t vcl_scalar_to_float(vcl_scalar_t const& vcl_s)
{
  cpu_scalar_t cpu_s = vcl_s;
  return cpu_s;
}

// Vector

/** @brief Returns a Python list describing the VCL_T */
bp::list vcl_vector_to_list(vcl_vector_t const& v)
{
  bp::list l;
  cpu_vector_t c(v.size());
  vcl::copy(v.begin(), v.end(), c.begin());
  
  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((vcl_vector_t::value_type::value_type)c[i]);
  
  return l;
}

np::ndarray vcl_vector_to_ndarray(vcl_vector_t const& v)
{
  return np::from_object(vcl_vector_to_list(v), np::dtype::get_builtin<cpu_scalar_t>());
}

/** @brief Creates the vector from the supplied ndarray */
vcl_vector_t vector_from_ndarray(np::ndarray const& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    bp::throw_error_already_set();
  }
  
  uint32_t s = (uint32_t) array.shape(0);
  
  vcl_vector_t v(s);
  cpu_vector_t cpu_vector(s);
  
  for (uint32_t i=0; i < s; ++i)
    cpu_vector[i] = bp::extract<cpu_scalar_t>(array[i]);
  
  vcl::fast_copy(cpu_vector.begin(), cpu_vector.end(), v.begin());

  return v;
}

/** @brief Creates the vector from the supplied Python list */
vcl_vector_t vector_from_list(bp::list const& l)
{
  return vector_from_ndarray(np::from_object(l, np::dtype::get_builtin<cpu_scalar_t>()));
}

vcl_vector_t new_scalar_vector(uint32_t length, cpu_scalar_t value) {
  return static_cast<vcl_vector_t>
    (vcl::scalar_vector<cpu_scalar_t>(length, value));
}

// Dense matrix

vcl_matrix_t new_scalar_matrix(uint32_t n, uint32_t m, cpu_scalar_t value) {
  return static_cast<vcl_matrix_t>
    (vcl::scalar_matrix<cpu_scalar_t>(n, m, value));
}

template<class ScalarT>
class ndarray_wrapper
{
  np::ndarray& array; // Reference to the wrapped ndarray

public:
  ndarray_wrapper(np::ndarray a)
    : array(a)
  { }

  uint32_t size1() const { return array.shape(0); }

  uint32_t size2() const { return array.shape(1); }

  ScalarT operator()(uint32_t row, uint32_t col) const
  {
    return bp::extract<ScalarT>(array[row][col]);
  } 

};

np::ndarray vcl_matrix_to_ndarray(vcl_matrix_t const& m)
{
  // Could generalise this for future tensor support, and work it into
  // the wrapper class above..

  uint32_t rows = m.size1();
  uint32_t cols = m.size2();

  // A less indirect method here would be to use vcl::backend::memory_read
  // to fill a chunk of memory that can be read as an np::ndarray..

  cpu_matrix_t cpu_m(rows, cols);

  vcl::copy(m, cpu_m);
 
  np::dtype dt = np::dtype::get_builtin<cpu_scalar_t>();
  bp::tuple shape = bp::make_tuple(rows, cols);

  np::ndarray array = np::empty(shape, dt);

  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      array[i][j] = cpu_m(i, j);
    }
  }

  return array;
}

/** @brief Creates the matrix from the supplied ndarray */
vcl_matrix_t matrix_from_ndarray(np::ndarray const& array)
{
  int d = array.get_nd();
  if (d != 2) {
    PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
    bp::throw_error_already_set();
  }
  
  ndarray_wrapper<cpu_scalar_t> wrapper(array);

  vcl_matrix_t mat(wrapper.size1(), wrapper.size2());

  vcl::copy(wrapper, mat);
  
  return mat;
}

// Sparse matrix

class cpu_sparse_matrix_wrapper
{
  uint32_t rows;

  // TODO: This is just a quick first implementation. Later, I may well want 
  // TODO: a version that doesn't depend on boost.python types, which are
  // TODO: somewhat untested relative to STL... (though they work well enough!)

  bp::list places;

public:
   cpu_sparse_matrix_t cpu_sparse_matrix;

  cpu_sparse_matrix_wrapper()
  {
    rows = 1;
    cpu_sparse_matrix = cpu_sparse_matrix_t(1);
  }

  cpu_sparse_matrix_wrapper(cpu_sparse_matrix_wrapper const& w)
  {
    rows = w.size1();
    cpu_sparse_matrix = w.cpu_sparse_matrix;
  }

  cpu_sparse_matrix_wrapper(uint32_t _rows)
    : rows(_rows)
  {
    cpu_sparse_matrix = cpu_sparse_matrix_t(rows);
  }

  cpu_sparse_matrix_wrapper(np::ndarray const& array)
  {

    // Doing things this way means we need at least 2x the amount of memory
    // required to store the data for the ndarray, just in order to construct
    // a sparse array on the compute device (which may itself take up
    // substantially less memory than the original densely stored ndarray..)
    // Ultimately, it would be better to wrap the ndarray with the features
    // required by vcl::copy -- but this is a good first go, in that it is
    // a straightforward implementation, and is functional enough to use
    // and for which to write regression tests.

    int d = array.get_nd();
    if (d != 2) {
      PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
      bp::throw_error_already_set();
    }
    
    uint32_t n = array.shape(0);
    uint32_t m = array.shape(1);
    
    cpu_sparse_matrix = cpu_sparse_matrix_t(n);
    rows = n;
    
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < m; ++j) {
	cpu_scalar_t val = bp::extract<cpu_scalar_t>(array[i][j]);
	if (val != 0) {
	  cpu_sparse_matrix[i][j] = val;
	  places.append(bp::make_tuple(i, j));
	}
      }
    }
    
  }

  uint32_t nnz()
  {
    
    uint32_t i = 0;  

    while (i < bp::len(places)) {
	
      bp::tuple item = bp::extract<bp::tuple>(places[i]);
      uint32_t n = bp::extract<uint32_t>(item[0]);
      uint32_t m = bp::extract<uint32_t>(item[1]);

      // We want to shift along the list. Conceptually, removing an item
      // has the same effect (for the "tape head") as increasing the index..
      if (cpu_sparse_matrix[n][m] == 0)
	places.remove(item);
      else
	++i;

    } 
      
    return bp::len(places);

  }

  uint32_t resize(uint32_t new_rows)
  {
    if (new_rows == rows)
      return rows;

    cpu_sparse_matrix.resize(new_rows);

    rows = new_rows;

    return rows;
  }

  uint32_t size1() const
  {
    // At the moment, this is inconsistent with size2() below. size2() returns
    // a position-relative figure (ie, the number of columns filled, not just
    // the absolute maximum abscissa), whereas this returns a position-absolute
    // figure (ie, the absolute maximum value of the ordinate).
    //
    // I probably should decide on a consistent scheme at some point.
    return rows;
  }

  uint32_t size2() const
  {
    uint32_t cols = 0;
    cpu_sparse_matrix_t::const_iterator iter = cpu_sparse_matrix.begin();

    for (uint32_t i = 0; iter < cpu_sparse_matrix.end(); ++i, ++iter) {
      if (cols < (*iter).size())
	cols = (*iter).size();
    }

    return cols;      
  }

  cpu_scalar_t& operator()(uint32_t n, uint32_t m)
  {
    if (n >= rows)
      resize(n+1);

    // We want to keep track of which places are filled.
    // If you access an unfilled location, then this increments the place list.
    // But the nnz() function checks for zeros at places referenced in that
    // list, so such increments don't matter, except for time wasted.
    bp::tuple loc = bp::make_tuple(n, m);
    if (not places.count(loc))
      places.append(loc);

    return cpu_sparse_matrix[n][m];
  }

  bp::object set(uint32_t n, uint32_t m, cpu_scalar_t val) 
  {
    (*this)(n, m) = val;
    return bp::object();
  }

  // Need this because bp cannot deal with operator()
  cpu_scalar_t get(uint32_t n, uint32_t m)
  {
    return (*this)(n, m);
  }

};


/*
*/

// cpu_sparse_matrix_from_ndarray

/*******************************
  Python module initialisation
 *******************************/

BOOST_PYTHON_MODULE(_viennacl)
{

  np::initialize();

  // --------------------------------------------------

  // *** Utility functions ***
  bp::def("backend_finish", vcl::backend::finish);

  // --------------------------------------------------

  // *** Scalar type ***

  bp::class_<vcl_scalar_t>("scalar")
    // Utility functions
    .def(bp::init<float>())
    .def(bp::init<int>())
    .def("get_value", &vcl_scalar_to_float)

    // Scalar-scalar operations
    .def("__add__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_add, 8>)
    .def("__add__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_add, 0>)

    .def("__sub__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_sub, 8>)
    .def("__sub__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_sub, 0>)

    .def("__mul__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_mul, 8>)
    .def("__mul__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_mul, 0>)

    .def("__truediv__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_div, 8>)
    .def("__truediv__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_div, 0>)

    // Scalar-vector operations
    .def("__mul__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_scalar_t&, vcl_vector_t&,
	 op_mul, 0>)

    // Scalar-matrix operations
    .def("__mul__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_scalar_t&, vcl_matrix_t&,
	 op_mul, 0>)

    ;

  // --------------------------------------------------

  // *** Vector type ***
  
  bp::class_<vcl_vector_t>("vector")
     .def(bp::init<int>())
     .def(bp::init<vcl_vector_t>())
    .def("get_value", &vcl_vector_to_ndarray)
    .def("clear", &vcl_vector_t::clear)
    //.def("resize", &vcl_vector_t::resize)
    .add_property("size", &vcl_vector_t::size)
    .add_property("internal_size", &vcl_vector_t::internal_size)

    // Basic arithmetic operations
    .def("__add__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_add, 0>)

    .def("__sub__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_sub, 0>)

    .def("__mul__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_scalar_t&,
	 op_mul, 0>)

    .def("__truediv__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_scalar_t&,
	 op_div, 0>)

    // In-place operations
    .def("__iadd__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_iadd, 0>)

    .def("__isub__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_isub, 0>)

    .def("__imul__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_scalar_t&,
	 op_imul, 0>)

    .def("__itruediv__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_scalar_t&,
	 op_idiv, 0>)

    // BLAS 1 not covered above
    .def("inner_prod", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_inner_prod, 0>)
    .def("dot", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_inner_prod, 0>)

    .def("element_prod", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_element_prod, 0>)
    .def("element_mul", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_element_prod, 0>)

    .def("element_div", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_element_div, 0>)
    .def("element_truediv", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_element_div, 0>)

    .add_property("norm_1", pyvcl_do_1ary_op<vcl_scalar_t,
		  vcl_vector_t&,
		  op_norm_1, 0>)
    .add_property("norm_2", pyvcl_do_1ary_op<vcl_scalar_t,
		  vcl_vector_t&,
		  op_norm_2, 0>)
    .add_property("norm_inf", pyvcl_do_1ary_op<vcl_scalar_t,
		  vcl_vector_t&,
		  op_norm_inf, 0>)
    .add_property("index_norm_inf", pyvcl_do_1ary_op<vcl_scalar_t,
		  vcl_vector_t&,
		  op_index_norm_inf, 0>)

    // BLAS 2
    .def("outer_prod", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_vector_t&, vcl_vector_t&,
	 op_outer_prod, 0>)
    ;

  bp::def("plane_rotation", pyvcl_do_4ary_op<bp::object,
	  vcl_vector_t&, vcl_vector_t&,
	  cpu_scalar_t, cpu_scalar_t,
	  op_plane_rotation, 0>);

  // *** Vector helper functions ***

  bp::def("vector_from_list", vector_from_list);
  bp::def("vector_from_ndarray", vector_from_ndarray);
  bp::def("scalar_vector", new_scalar_vector);

  // --------------------------------------------------

  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  // *** Dense matrix type ***

  bp::class_<vcl_matrix_t>("matrix")
    .def(bp::init<vcl_matrix_t>())
    .def(bp::init<int, int>())

    .def("get_value", &vcl_matrix_to_ndarray)
    .def("clear", &vcl_matrix_t::clear)
    //.def("resize", &vcl_matrix_t::resize)

    .add_property("size1", &vcl_matrix_t::size1)
    .add_property("internal_size1", &vcl_matrix_t::internal_size1)
    .add_property("size2", &vcl_matrix_t::size2)
    .add_property("internal_size2", &vcl_matrix_t::internal_size2)

    .add_property("trans", pyvcl_do_1ary_op<vcl_matrix_t,
		  vcl_matrix_t&,
		  op_trans, 0>)

    .def("__add__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_matrix_t&,
	 op_add, 0>)

    .def("__sub__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_matrix_t&,
	 op_sub, 0>)

    .def("__mul__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_scalar_t&,
	 op_mul, 0>)

    .def("__truediv__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_scalar_t&,
	 op_div, 0>)

    .def("__iadd__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_matrix_t&,
	 op_iadd, 0>)

    .def("__isub__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_matrix_t&,
	 op_isub, 0>)

    .def("__imul__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_scalar_t&,
	 op_imul, 0>)

    .def("__itruediv__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t&, vcl_scalar_t&,
	 op_idiv, 0>)

    .def("prod", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_matrix_t, vcl_vector_t,
	 op_prod, 0>)
    .def("prod", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_matrix_t, vcl_matrix_t,
	 op_prod, 0>)

    .def("solve", pyvcl_do_3ary_op<vcl_vector_t,
	 vcl_matrix_t, vcl_vector_t,
	 vcl::linalg::lower_tag,
	 op_solve, 0>)
    .def("solve", pyvcl_do_3ary_op<vcl_vector_t,
	 vcl_matrix_t, vcl_vector_t,
	 vcl::linalg::unit_lower_tag,
	 op_solve, 0>)
    .def("solve", pyvcl_do_3ary_op<vcl_vector_t,
	 vcl_matrix_t, vcl_vector_t,
	 vcl::linalg::upper_tag,
	 op_solve, 0>)
    .def("solve", pyvcl_do_3ary_op<vcl_vector_t,
	 vcl_matrix_t, vcl_vector_t,
	 vcl::linalg::unit_upper_tag,
	 op_solve, 0>)

    .def("solve", pyvcl_do_3ary_op<vcl_matrix_t,
	 vcl_matrix_t, vcl_matrix_t,
	 vcl::linalg::lower_tag,
	 op_solve, 0>)
    .def("solve", pyvcl_do_3ary_op<vcl_matrix_t,
	 vcl_matrix_t, vcl_matrix_t,
	 vcl::linalg::unit_lower_tag,
	 op_solve, 0>)
    .def("solve", pyvcl_do_3ary_op<vcl_matrix_t,
	 vcl_matrix_t, vcl_matrix_t,
	 vcl::linalg::upper_tag,
	 op_solve, 0>)
    .def("solve", pyvcl_do_3ary_op<vcl_matrix_t,
	 vcl_matrix_t, vcl_matrix_t,
	 vcl::linalg::unit_upper_tag,
	 op_solve, 0>)
    ;
           
  // *** Dense matrix helper functions ***

  bp::def("matrix_from_ndarray", matrix_from_ndarray);
  bp::def("scalar_matrix", new_scalar_matrix);

  // --------------------------------------------------

  // *** Sparse matrix types ***

  bp::class_<cpu_sparse_matrix_wrapper>("cpu_sparse_matrix")
    .def(bp::init<>())
    .def(bp::init<uint32_t>())
    .def(bp::init<cpu_sparse_matrix_wrapper>())
    .def(bp::init<np::ndarray>())
    .add_property("nnz", &cpu_sparse_matrix_wrapper::nnz)
    .add_property("size1", &cpu_sparse_matrix_wrapper::size1)
    .add_property("size2", &cpu_sparse_matrix_wrapper::size2)
    .def("set", &cpu_sparse_matrix_wrapper::set)
    .def("get", &cpu_sparse_matrix_wrapper::get)
    ;

  bp::class_<vcl_compressed_matrix_t>("compressed_matrix")
    ;

  // *** Sparse matrix helper functions ***

  // ...

  /*

    TODO:
    + cpu_sparse_matrix.as_ndarray
    + vcl sparse matrix types

   */

}

