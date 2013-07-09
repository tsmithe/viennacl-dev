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
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>
#include <cstdint>
#include <iostream>
#include <typeinfo>

namespace vcl = viennacl;
namespace bp = boost::python;
namespace np = boost::numpy;
namespace ublas = boost::numeric::ublas;

typedef vcl::scalar<double> vcl_scalar_t;
typedef double              cpu_scalar_t;

typedef std::vector<cpu_scalar_t> cpu_vector_t;
typedef vcl::vector<cpu_scalar_t> vcl_vector_t;

typedef vcl::matrix<cpu_scalar_t,
		    vcl::row_major> vcl_matrix_t;
typedef ublas::matrix<cpu_scalar_t,
		  ublas::row_major> cpu_matrix_t;

typedef void* NoneT;

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

/*******************************
  Arithmetic wrapper functions
 *******************************/

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
struct pyvcl_worker
{
  static ReturnT do_op(void* o) {}
};

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
      
      The value of PyObjs determines which operands need to be extracted from
      Python objects, by coding the operand "position" in binary.
      
      So, given PyObjs == 7 == 0111b, and given that we number operands from
      left to right, the following operands need extraction: operand2,
      operand3, and operand4.
      
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


template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT,
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_add, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_add, PyObjs>& o)
  {
    return o.operand1 + o.operand2;
  }
};
  
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_sub, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_sub, PyObjs>& o)
  {
    return o.operand1 - o.operand2;
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_mul, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_mul, PyObjs>& o)
  {
    return o.operand1 * o.operand2;
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_div, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_div, PyObjs>& o)
  {
    return o.operand1 / o.operand2;
  }
};


template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,		    
		    op_inner_prod, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_inner_prod, PyObjs>& o)
  {
    return vcl::linalg::inner_prod(o.operand1, o.operand2);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,		    
		    op_outer_prod, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_outer_prod, PyObjs>& o)
  {
    return vcl::linalg::outer_prod(o.operand1, o.operand2);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_element_prod, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_element_prod, PyObjs>& o)
  {
    return vcl::linalg::element_prod(o.operand1, o.operand2);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_element_div, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_element_div, PyObjs>& o)
  {
    return vcl::linalg::element_div(o.operand1, o.operand2);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_iadd, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_iadd, PyObjs>& o)
  {
    o.operand1 += o.operand2;
    return o.operand1;
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_isub, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_isub, PyObjs>& o)
  {
    o.operand1 -= o.operand2;
    return o.operand1;
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_imul, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_imul, PyObjs>& o)
  {
    o.operand1 *= o.operand2;
    return o.operand1;
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_idiv, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_idiv, PyObjs>& o)
  {
    o.operand1 /= o.operand2;
    return o.operand1;
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_norm_1, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_norm_1, PyObjs>& o)
  {
    return vcl::linalg::norm_1(o.operand1);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_norm_2, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_norm_2, PyObjs>& o)
  {
    return vcl::linalg::norm_2(o.operand1);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_norm_inf, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT,
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,
		       op_norm_inf, PyObjs>& o)
  {
    return vcl::linalg::norm_inf(o.operand1);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_index_norm_inf, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,		       
		       op_index_norm_inf, PyObjs>& o)
  {
    return vcl::linalg::index_norm_inf(o.operand1);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_plane_rotation, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,		       
		       op_plane_rotation, PyObjs>& o)
  {
    vcl::linalg::plane_rotation(o.operand1, o.operand2,
				o.operand3, o.operand4);
    return bp::object();
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_trans, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,		       
		       op_trans, PyObjs>& o)
  {
    return vcl::trans(o.operand1);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_prod, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,		       
		       op_prod, PyObjs>& o)
  {
    return vcl::linalg::prod(o.operand1, o.operand2);
  }
};

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  int PyObjs>
struct pyvcl_worker<ReturnT, 
		    Operand1T, Operand2T,
		    Operand3T, Operand4T,
		    op_solve, PyObjs>
{
  static ReturnT do_op(pyvcl_op<ReturnT, 
		       Operand1T, Operand2T,
		       Operand3T, Operand4T,		       
		       op_solve, PyObjs>& o)
  {
    return vcl::linalg::solve(o.operand1, o.operand2,
			      o.operand3);
  }
};

// Worker functions

template <class ReturnT,
	  class Operand1T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_1ary_op(Operand1T a)
{
  pyvcl_op<ReturnT,
	   Operand1T, NoneT,
	   NoneT, NoneT,
	   op, PyObjs>
    o = pyvcl_op<ReturnT,
		 Operand1T, NoneT,
		 NoneT, NoneT,
		 op, PyObjs>(a, NULL, NULL, NULL);
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
    o(a,b,NULL,NULL);/* = pyvcl_op<ReturnT,
		 Operand1T, Operand2T,
		 NoneT, NoneT,
		 op, PyObjs>(a, b, NULL, NULL);*/
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
    o = pyvcl_op<ReturnT,
		 Operand1T, Operand2T,
		 Operand3T, NoneT,
		 op, PyObjs>(a, b, c, NULL);
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
      o = pyvcl_op<ReturnT,
		 Operand1T, Operand2T,
		 Operand3T, Operand4T,
		 op, PyObjs>(a, b, c, d);
  return o.do_op();
}


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

/** @brief Creates the vector from the supplied ndarray */
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
    //.def("__add__", pyvcl_do_2ary_op<vcl_scalar_t, vcl_scalar_t&, cpu_scalar_t&, op_add, 1>)

    .def("__sub__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_sub, 8>)
    .def("__sub__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_sub, 0>)
    //.def("__sub__", pyvcl_do_2ary_op<vcl_scalar_t, vcl_scalar_t, cpu_scalar_t, op_sub, 1>)

    .def("__mul__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_mul, 8>)
    .def("__mul__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_mul, 0>)
    //.def("__mul__", pyvcl_do_2ary_op<vcl_scalar_t, vcl_scalar_t, cpu_scalar_t, op_mul, 1>)

    .def("__truediv__", pyvcl_do_2ary_op<vcl_scalar_t,
	 cpu_scalar_t&, vcl_scalar_t&,
	 op_div, 8>)
    .def("__truediv__", pyvcl_do_2ary_op<vcl_scalar_t,
	 vcl_scalar_t&, vcl_scalar_t&,
	 op_div, 0>)
    //.def("__truediv__", pyvcl_do_2ary_op<vcl_scalar_t, vcl_scalar_t&, cpu_scalar_t&, op_div, 1>)

    //.def("__floordiv__", ...)
    //.def("__pow__", ...)
    //.def("__ifloordiv__", ...)
    //.def("__ipow__", ...)

    // Scalar-vector operations
    .def("__mul__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_scalar_t&, vcl_vector_t&,
	 op_mul, 0>)

    // Scalar-matrix operations
    .def("__mul__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_scalar_t&, vcl_matrix_t&,
	 op_mul, 0>)

    /*

      TODO:
      + assignment to/from Python float
      + comparison operators
      + addition, multiplication, exponentiation, subtraction
      + division (quotient and integer quotient), modulus, divmod
      + truncation, rounding, floor, ceil
      + as_integer_ratio
      + is_integer
      + hex/fromhex

      Which of these should be implemented here?
      Which in the Python wrapper?

     */
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
    //.def("__mul__", vcl_op_obj_l<vcl_vector_t, cpu_scalar_t, vcl_vector_t const&>, op_mul)
    //.def("__mul__", vcl_op_obj_r<vcl_vector_t, cpu_scalar_t, vcl_vector_t const&>, op_mul)

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
    /*

      TODO:
      + clear, resize, size, internal_size, swap, empty, handle
      + BLAS level 1 basic arithmetic operations

     */
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
           
  // *** Matrix helper functions **

  bp::def("matrix_from_ndarray", matrix_from_ndarray);
  bp::def("scalar_matrix", new_scalar_matrix);

}

