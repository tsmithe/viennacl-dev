#include <stdint.h>

#include <iostream>
#include <typeinfo>

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>

#define VIENNACL_WITH_UBLAS
//#define VIENNACL_WITH_OPENCL
//#define VIENNACL_WITH_PYTHON
#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/sparse_matrix_operations.hpp>
#include <viennacl/compressed_matrix.hpp>
//#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/scheduler/execute.hpp>

namespace vcl = viennacl;
namespace bp = boost::python;
namespace np = boost::numpy;
namespace ublas = boost::numeric::ublas;

typedef void* NoneT;

typedef vcl::scalar<double> vcl_scalar_t;
typedef double              cpu_scalar_t;

// Dense types

typedef vcl::matrix<cpu_scalar_t,
		    vcl::row_major> vcl_matrix_t;
typedef ublas::matrix<cpu_scalar_t,
		  ublas::row_major> cpu_matrix_t;

// Sparse types

typedef ublas::compressed_vector<cpu_scalar_t> cpu_sparse_vector_t;

/* Would like to use this, since it's faster than a uBLAS compressed matrix, 
   but it seems to give some sort of type-ambiguity error right now..
typedef ublas::generalized_vector_of_vector<cpu_scalar_t,
					    ublas::row_major,
					    ublas::vector<cpu_sparse_vector_t> >
cpu_sparse_matrix_t;
*/

// Originally used a vector-map implementation, but that's not always defined
typedef ublas::compressed_matrix<cpu_scalar_t,
				 ublas::row_major> cpu_sparse_matrix_t;

typedef vcl::compressed_matrix<cpu_scalar_t> vcl_compressed_matrix_t;
//typedef vcl::coordinate_matrix<cpu_scalar_t> vcl_coordinate_matrix_t;

typedef vcl::ell_matrix<cpu_scalar_t> vcl_ell_matrix_t;
typedef vcl::hyb_matrix<cpu_scalar_t> vcl_hyb_matrix_t;

enum op_t {
  op_add,              // done in sched: vector, ...
  op_sub,              // done in sched: vector, ..
  op_mul,
  op_div,              // done in sched: ..
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
  op_solve,
  op_inplace_solve
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
//
// Ultimately, I may well do away with this, and interface with the kernel
// scheduler directly. But this is a useful start to have, in order to get
// a working prototype.
//
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

DO_OP_FUNC(op_inplace_solve)
{
  vcl::linalg::inplace_solve(o.operand1, o.operand2,
			     o.operand3);
  return o.operand1;
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

template <class SCALARTYPE>
bp::list vcl_vector_to_list(vcl::vector<SCALARTYPE> const& v)
{
  bp::list l;
  std::vector<SCALARTYPE> c(v.size());
  vcl::copy(v.begin(), v.end(), c.begin());
  
  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((SCALARTYPE)c[i]);
  
  return l;
}

template <class SCALARTYPE>
np::ndarray vcl_vector_to_ndarray(vcl::vector<SCALARTYPE> const& v)
{
  return np::from_object(vcl_vector_to_list<SCALARTYPE>(v),
			 np::dtype::get_builtin<SCALARTYPE>());
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector<SCALARTYPE> >
vector_init_ndarray(np::ndarray const& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    bp::throw_error_already_set();
  }
  
  uint32_t s = (uint32_t) array.shape(0);
  
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(s);
  std::vector<SCALARTYPE> cpu_vector(s);
  
  for (uint32_t i=0; i < s; ++i)
    cpu_vector[i] = bp::extract<SCALARTYPE>(array[i]);
  
  vcl::fast_copy(cpu_vector.begin(), cpu_vector.end(), v->begin());

  return boost::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector<SCALARTYPE> >
vector_init_list(bp::list const& l)
{
  return vector_init_ndarray<SCALARTYPE>
    (np::from_object(l, np::dtype::get_builtin<SCALARTYPE>()));
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector<SCALARTYPE> >
vector_init_scalar(uint32_t length, SCALARTYPE value) {
  ublas::scalar_vector<SCALARTYPE> s_v(length, value);
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(length);
  vcl::copy(s_v.begin(), s_v.end(), v->begin());
  return boost::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

// Dense matrix

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

boost::shared_ptr<vcl_matrix_t>
matrix_init_scalar(uint32_t n, uint32_t m, cpu_scalar_t value) {
  ublas::scalar_matrix<cpu_scalar_t> s_m(n, m, value);
  vcl_matrix_t* mat = new vcl_matrix_t(n, m);
  vcl::copy(s_m, (*mat));
  return boost::shared_ptr<vcl_matrix_t>(mat);
}

/** @brief Creates the matrix from the supplied ndarray */
boost::shared_ptr<vcl_matrix_t>
matrix_init_ndarray(np::ndarray const& array)
{
  int d = array.get_nd();
  if (d != 2) {
    PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
    bp::throw_error_already_set();
  }
  
  ndarray_wrapper<cpu_scalar_t> wrapper(array);

  vcl_matrix_t* mat = new vcl_matrix_t(wrapper.size1(), wrapper.size2());

  vcl::copy(wrapper, (*mat));
  
  return boost::shared_ptr<vcl_matrix_t>(mat);
}

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

// Sparse matrix

class cpu_sparse_matrix_wrapper
{
  // TODO: This is just a quick first implementation. Later, I may well want 
  // TODO: a version that doesn't depend on boost.python types.
  bp::list places;

public:
   cpu_sparse_matrix_t cpu_sparse_matrix;

  bp::list const& update_places()
  {

    for (uint32_t i = 0; i < size1(); ++i) {
      for (uint32_t j = 0; j < size2(); ++j) {
	if (cpu_sparse_matrix(i, j) != 0) {
	  places.append(bp::make_tuple(i, j));
	}
      }
    }

    return places;

  }

  cpu_sparse_matrix_wrapper()
  {
    cpu_sparse_matrix = cpu_sparse_matrix_t(0,0,0);
  }

  cpu_sparse_matrix_wrapper(uint32_t _size1, uint32_t _size2)
  {
    cpu_sparse_matrix = cpu_sparse_matrix_t(_size1, _size2);
  }

  cpu_sparse_matrix_wrapper(uint32_t _size1, uint32_t _size2, uint32_t _nnz)
  {
    cpu_sparse_matrix = cpu_sparse_matrix_t(_size1, _size2, _nnz);
  }

  cpu_sparse_matrix_wrapper(cpu_sparse_matrix_wrapper const& w)
    : cpu_sparse_matrix(w.cpu_sparse_matrix)
  {
    update_places();
  }

  template<class SparseT>
  cpu_sparse_matrix_wrapper(SparseT const& vcl_sparse_matrix)
  {
    cpu_sparse_matrix = cpu_sparse_matrix_t(vcl_sparse_matrix.size1(),
					    vcl_sparse_matrix.size2());
    vcl::copy(vcl_sparse_matrix, cpu_sparse_matrix);
    
    update_places();
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
    
    cpu_sparse_matrix = cpu_sparse_matrix_t(n, m);
    
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < m; ++j) {
	cpu_scalar_t val = bp::extract<cpu_scalar_t>(array[i][j]);
	if (val != 0) {
	  cpu_sparse_matrix(i, j) = val;
	  places.append(bp::make_tuple(i, j));
	}
      }
    }
    
  }

  np::ndarray as_ndarray()
  {

    np::dtype dt = np::dtype::get_builtin<cpu_scalar_t>();
    bp::tuple shape = bp::make_tuple(size1(), size2());
    
    np::ndarray array = np::empty(shape, dt);
  
    for (uint32_t i = 0; i < size1(); ++i) {
      for (uint32_t j = 0; j < size2(); ++j) {
	array[i][j] = (cpu_scalar_t) cpu_sparse_matrix(i, j);
      }
    }

    return array;

  }

  template<class SparseT>
  SparseT as_vcl_sparse_matrix()
  {
    SparseT vcl_sparse_matrix;
    vcl::copy(cpu_sparse_matrix, vcl_sparse_matrix);
    return vcl_sparse_matrix;
  }

  template<class SparseT>
  SparseT as_vcl_sparse_matrix_with_size()
  {
    SparseT vcl_sparse_matrix(size1(), size2(), nnz());
    vcl::copy(cpu_sparse_matrix, vcl_sparse_matrix);
    return vcl_sparse_matrix;
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
      if (cpu_sparse_matrix(n, m) == 0)
	places.remove(item);
      else
	++i;

    } 
      
    return bp::len(places);

  }

  bp::object resize(uint32_t _size1, uint32_t _size2)
  {
  
    if ((_size1 == size1()) and (_size2 == size2()))
      return bp::object();

    cpu_sparse_matrix.resize(_size1, _size2, false);

    return bp::object();
  }

  uint32_t size1() const
  {
    return cpu_sparse_matrix.size1();
  }

  uint32_t size2() const
  {
    return cpu_sparse_matrix.size2();
  }

  bp::object set(uint32_t n, uint32_t m, cpu_scalar_t val) 
  {
    if (n >= size1() or m >= size2())
      resize(n+1, m+1);

    // We want to keep track of which places are filled.
    // If you access an unfilled location, then this increments the place list.
    // But the nnz() function checks for zeros at places referenced in that
    // list, so such increments don't matter, except for time wasted.
    bp::tuple loc = bp::make_tuple(n, m);
    if (not places.count(loc))
      places.append(loc);

    cpu_sparse_matrix(n, m) = val;
    return bp::object();
  }

  // Need this because bp cannot deal with operator()
  cpu_scalar_t get(uint32_t n, uint32_t m)
  {
    return cpu_sparse_matrix(n, m);
  }

};

/************************************************
        Scheduler / generator interface  
 ************************************************/

/*

  Want to wrap the expression node class viennacl::scheduler::statement_node.
  The expression tree is a connected graph of nodes of this class.

  In principle, the nodes of the graph do not need to occupy a contiguous
  region of memory. We ought to be able just to construct nodes arbitrarily
  and use pointers. This would also make it very easy to modify the
  expression graph post hoc.

  Currently, though, this behaviour is restricted: the nodes must form the
  elements of a vector, and references are given not as pointers but as
  indices on that vector. As far as I can tell, this is an implementation
  detail, and should be open to generalisation.

  So, for now, we also need a function to create such a vector, returning type
   viennacl::scheduler::statement::container_type.

  Then, wrap viennacl::scheduler::statement such that the CTOR takes a graph
  or vector of nodes to a statement instance.

  Lastly for now, need to wrap viennacl::scheduler::execute(...), in order
  to do any work.

*/

class statement_node_wrapper {

  //typedef std::size_t node_index;
  vcl::scheduler::statement::value_type vcl_node;

public:

  statement_node_wrapper(const statement_node_wrapper& node)
    : vcl_node(node.vcl_node)
  { }

  statement_node_wrapper(vcl::scheduler::statement_node node)
    : vcl_node(node)
  { }

  statement_node_wrapper(vcl::scheduler::statement_node_type_family lhs_family,
			 vcl::scheduler::statement_node_type lhs_type,
			 vcl::scheduler::operation_node_type_family op_family,
			 vcl::scheduler::operation_node_type op_type,
			 vcl::scheduler::statement_node_type_family rhs_family,
			 vcl::scheduler::statement_node_type rhs_type)
  {
    vcl_node.op.type_family = op_family;
    vcl_node.op.type = op_type;
    vcl_node.lhs.type_family = lhs_family;
    vcl_node.lhs.type = lhs_type;
    vcl_node.rhs.type_family = rhs_family;
    vcl_node.rhs.type = rhs_type;
  }

  /*
  vcl::scheduler::statement_node& get_vcl_statement_node()
  {
    return vcl_node;
  }
  */

  vcl::scheduler::statement_node get_vcl_statement_node() const
  {
    return vcl_node;
  }

  /*
  void set_operand_to_node_index(uint8_t operand, std::size_t i)
  {
    switch (operand) {
    case 0:
      vcl_node.lhs.node_index = i;
      break;
    case 1:
      vcl_node.rhs.node_index = i;
      break;
    }
  }
  */

#define CONCAT(...) __VA_ARGS__

#define SET_OPERAND(T, I)					   \
  void set_operand_to_ ## I (int o, T I) {			   \
    switch (o) {						   \
    case 0:							   \
      vcl_node.lhs.I  = I;					   \
      break;							   \
    case 1:							   \
      vcl_node.rhs.I  = I;					   \
      break;							   \
    default:							   \
      throw viennacl::scheduler::statement_not_supported_exception \
	("Only support operands 0 or 1");			   \
    }								   \
  }

  SET_OPERAND(std::size_t,       node_index)

  SET_OPERAND(char,              host_char)
  SET_OPERAND(unsigned char,     host_uchar)
  SET_OPERAND(short,             host_short)
  SET_OPERAND(unsigned short,    host_ushort)
  SET_OPERAND(int,               host_int)
  SET_OPERAND(unsigned int,      host_uint)
  SET_OPERAND(long,              host_long)
  SET_OPERAND(unsigned long,     host_ulong)
  SET_OPERAND(float,             host_float)
  SET_OPERAND(double,            host_double)

  // NB: need to add remaining scalar types as they become available
  SET_OPERAND(vcl::scalar<float>*, scalar_float)
  SET_OPERAND(vcl::scalar<double>*, scalar_double)

  // NB: need to add remaining vector types as they become available
  SET_OPERAND(vcl::vector<float>*, vector_float)
  SET_OPERAND(vcl::vector<double>*, vector_double)

  // NB: need to add remaining matrix_row types as they become available
  SET_OPERAND(vcl::matrix<float>*, matrix_row_float)
  SET_OPERAND(vcl::matrix<double>*, matrix_row_double)
  
  // NB: need to add remaining matrix_col types as they become available
  SET_OPERAND(CONCAT(vcl::matrix_base<float, viennacl::column_major>*),
    matrix_col_float)
  SET_OPERAND(CONCAT(vcl::matrix_base<double, viennacl::column_major>*),
    matrix_col_double)

};
#undef SET_OPERAND

class statement_wrapper {
  typedef vcl::scheduler::statement::container_type nodes_container_t;

  typedef nodes_container_t::iterator nodes_iterator;
  typedef nodes_container_t::const_iterator nodes_const_iterator;
  
  nodes_container_t vcl_expression_nodes;

public:

  statement_wrapper() {
    vcl_expression_nodes = nodes_container_t(0);
  }

  void execute() {
    vcl::scheduler::statement tmp_statement(vcl_expression_nodes);
    vcl::scheduler::execute(tmp_statement);
  }

  std::size_t size() const {
    return vcl_expression_nodes.size();
  }

  void clear() {
    vcl_expression_nodes.clear();
  }
  
  statement_node_wrapper get_node(std::size_t offset) const {
    return statement_node_wrapper(vcl_expression_nodes[offset]);
  }

  void erase_node(std::size_t offset)
  {
    nodes_iterator it = vcl_expression_nodes.begin();
    vcl_expression_nodes.erase(it+offset);
  }

  void insert_at_index(std::size_t offset,
		       const statement_node_wrapper& node)
  {
    nodes_iterator it = vcl_expression_nodes.begin();
    vcl_expression_nodes.insert(it+offset, node.get_vcl_statement_node());
  }

  void insert_at_begin(const statement_node_wrapper& node)
  {
    nodes_iterator it = vcl_expression_nodes.begin();
    vcl_expression_nodes.insert(it, node.get_vcl_statement_node());
  }

  void insert_at_end(const statement_node_wrapper& node)
  {
    vcl_expression_nodes.push_back(node.get_vcl_statement_node());
  }

};


/*******************************
  Python module initialisation
 *******************************/

/*
void translate_scheduler_exception(vcl::scheduler::statement_not_supported_exception e)
{
  // Use the Python 'C' API to set up an exception object
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
*/

BOOST_PYTHON_MODULE(_viennacl)
{

  /*
  bp::register_exception_translator
    <vcl::scheduler::statement_not_supported_exception>
    (&translate_scheduler_exception);
  */

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
    .def("as_double", &vcl_scalar_to_float)

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

    /*
    // Scalar-vector operations
    .def("__mul__", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_scalar_t&, vcl_vector_t&,
	 op_mul, 0>)
    */

    // Scalar-matrix operations
    .def("__mul__", pyvcl_do_2ary_op<vcl_matrix_t,
	 vcl_scalar_t&, vcl_matrix_t&,
	 op_mul, 0>)

    ;

  // --------------------------------------------------

  // *** Vector types ***
  
#define EXPORT_VECTOR_CLASS(TYPE, NAME)					\
  bp::class_<vcl::vector<TYPE>,						\
	     boost::shared_ptr<vcl::vector<TYPE> > >			\
    ( NAME )								\
    .def(bp::init<int>())						\
    .def(bp::init<vcl::vector<TYPE> >())				\
    .def("__init__", bp::make_constructor(vector_init_ndarray<TYPE>))	\
    .def("__init__", bp::make_constructor(vector_init_list<TYPE>))	\
    .def("__init__", bp::make_constructor(vector_init_scalar<TYPE>))	\
    .def("as_ndarray", &vcl_vector_to_ndarray<TYPE>)			\
    .def("clear", &vcl::vector<TYPE>::clear)				\
    .add_property("size", &vcl::vector<TYPE>::size)			\
    .add_property("internal_size", &vcl::vector<TYPE>::internal_size)	\
    .add_property("norm_1", pyvcl_do_1ary_op<vcl::scalar<TYPE>,		\
		  vcl::vector<TYPE>&,					\
		  op_norm_1, 0>)					\
    .add_property("norm_2", pyvcl_do_1ary_op<vcl::scalar<TYPE>,		\
		  vcl::vector<TYPE>&,					\
		  op_norm_2, 0>)					\
    .add_property("norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,	\
		  vcl::vector<TYPE>&,					\
		  op_norm_inf, 0>)					\
    .add_property("index_norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,	\
		  vcl::vector<TYPE>&,					\
		  op_index_norm_inf, 0>)				\
    .def("__add__", pyvcl_do_2ary_op<vcl::vector<TYPE>,			\
	 vcl::vector<TYPE>&, vcl::vector<TYPE>&,			\
	 op_add, 0>)							\
    ;

  EXPORT_VECTOR_CLASS(float, "vector_float")
  EXPORT_VECTOR_CLASS(double, "vector_double")

  /*
  bp::def("plane_rotation", pyvcl_do_4ary_op<bp::object,
	  vcl_vector_t&, vcl_vector_t&,
	  cpu_scalar_t, cpu_scalar_t,
	  op_plane_rotation, 0>);
  */

  // --------------------------------------------------

  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  // *** Dense matrix type ***

  bp::class_<vcl_matrix_t, boost::shared_ptr<vcl_matrix_t> >("matrix")
    .def(bp::init<vcl_matrix_t>())
    .def(bp::init<uint32_t, uint32_t>())
    .def("__init__", bp::make_constructor(matrix_init_ndarray))
    .def("__init__", bp::make_constructor(matrix_init_scalar))

    .def("as_ndarray", &vcl_matrix_to_ndarray)
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

    /*
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
    */

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
           
  // --------------------------------------------------

  // *** Sparse matrix types ***

  bp::class_<cpu_sparse_matrix_wrapper>("cpu_sparse_matrix")
    .def(bp::init<>())
    .def(bp::init<uint32_t, uint32_t>())
    .def(bp::init<uint32_t, uint32_t, uint32_t>())
    .def(bp::init<cpu_sparse_matrix_wrapper>())
    .def(bp::init<vcl_compressed_matrix_t>())
    //.def(bp::init<vcl_coordinate_matrix_t>())
    .def(bp::init<vcl_ell_matrix_t>())
    .def(bp::init<vcl_hyb_matrix_t>())
    .def(bp::init<np::ndarray>())
    .add_property("nnz", &cpu_sparse_matrix_wrapper::nnz)
    .add_property("size1", &cpu_sparse_matrix_wrapper::size1)
    .add_property("size2", &cpu_sparse_matrix_wrapper::size2)
    .def("set", &cpu_sparse_matrix_wrapper::set)
    .def("get", &cpu_sparse_matrix_wrapper::get)
    .def("as_ndarray", &cpu_sparse_matrix_wrapper::as_ndarray)
    .def("as_compressed_matrix",
	 &cpu_sparse_matrix_wrapper::as_vcl_sparse_matrix_with_size
	 <vcl_compressed_matrix_t>)
    /*.def("as_coordinate_matrix",
	 &cpu_sparse_matrix_wrapper::as_vcl_sparse_matrix
	 <vcl_coordinate_matrix_t>)*/
    .def("as_ell_matrix",
	 &cpu_sparse_matrix_wrapper::as_vcl_sparse_matrix
	 <vcl_ell_matrix_t>)
    .def("as_hyb_matrix",
	 &cpu_sparse_matrix_wrapper::as_vcl_sparse_matrix
	 <vcl_hyb_matrix_t>)
    ;

  bp::class_<vcl_compressed_matrix_t>("compressed_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl_compressed_matrix_t::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl_compressed_matrix_t::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("nnz",
		  make_function(&vcl_compressed_matrix_t::nnz,
			      bp::return_value_policy<bp::return_by_value>()))

    /*
    .def("prod", pyvcl_do_2ary_op<vcl_vector_t,
	 vcl_compressed_matrix_t, vcl_vector_t,
	 op_prod, 0>)

    .def("inplace_solve", pyvcl_do_3ary_op<vcl_compressed_matrix_t,
	 vcl_compressed_matrix_t&, vcl_vector_t&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl_compressed_matrix_t,
	 vcl_compressed_matrix_t&, vcl_vector_t&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl_compressed_matrix_t,
	 vcl_compressed_matrix_t&, vcl_vector_t&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl_compressed_matrix_t,
	 vcl_compressed_matrix_t&, vcl_vector_t&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve, 0>)
    */

    /*.def("solve", pyvcl_do_3ary_op<vcl_matrix_t,
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
	 op_solve, 0>)*/
    ;

  /*
  bp::class_<vcl_coordinate_matrix_t, boost::noncopyable>("coordinate_matrix")
    ;
  */

  /*
  bp::class_<vcl_ell_matrix_t>("ell_matrix")
    .def(bp::init<>())
    ;
  */

  // Scheduler interface (first attempt)

#define VALUE(NS, V) .value( #V, NS :: V )

  bp::enum_<vcl::scheduler::operation_node_type_family>
    ("operation_node_type_family")
    VALUE(vcl::scheduler, OPERATION_UNARY_TYPE_FAMILY)
    VALUE(vcl::scheduler, OPERATION_BINARY_TYPE_FAMILY)
    ;

  bp::enum_<vcl::scheduler::operation_node_type>("operation_node_type")
    // unary expression
    VALUE(vcl::scheduler, OPERATION_UNARY_ABS_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_ACOS_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_ASIN_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_ATAN_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_CEIL_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_COS_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_COSH_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_EXP_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_FABS_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_FLOOR_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_LOG_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_LOG10_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_SIN_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_SINH_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_SQRT_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_TAN_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_TANH_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_NORM_1_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_NORM_2_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_NORM_INF_TYPE)
    
    // binary expression
    VALUE(vcl::scheduler, OPERATION_BINARY_ASSIGN_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_INPLACE_ADD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_INPLACE_SUB_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_ADD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_SUB_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_MAT_VEC_PROD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_MAT_MAT_PROD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_MULT_TYPE)// scalar*vector/matrix
    VALUE(vcl::scheduler, OPERATION_BINARY_ELEMENT_MULT_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_ELEMENT_DIV_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_INNER_PROD_TYPE)
    ;

  bp::enum_<vcl::scheduler::statement_node_type_family>
    ("statement_node_type_family")
    VALUE(vcl::scheduler, COMPOSITE_OPERATION_FAMILY)
    VALUE(vcl::scheduler, HOST_SCALAR_TYPE_FAMILY)
    VALUE(vcl::scheduler, SCALAR_TYPE_FAMILY)
    VALUE(vcl::scheduler, VECTOR_TYPE_FAMILY)
    VALUE(vcl::scheduler, MATRIX_ROW_TYPE_FAMILY)
    VALUE(vcl::scheduler, MATRIX_COL_TYPE_FAMILY)
    ;

  bp::enum_<vcl::scheduler::statement_node_type>("statement_node_type")
    VALUE(vcl::scheduler, COMPOSITE_OPERATION_TYPE)

    // host scalars:
    VALUE(vcl::scheduler, HOST_SCALAR_CHAR_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_UCHAR_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_SHORT_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_USHORT_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_INT_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_UINT_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_LONG_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_ULONG_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_HALF_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_FLOAT_TYPE)
    VALUE(vcl::scheduler, HOST_SCALAR_DOUBLE_TYPE)
    
    // device scalars:
    VALUE(vcl::scheduler, SCALAR_CHAR_TYPE)
    VALUE(vcl::scheduler, SCALAR_UCHAR_TYPE)
    VALUE(vcl::scheduler, SCALAR_SHORT_TYPE)
    VALUE(vcl::scheduler, SCALAR_USHORT_TYPE)
    VALUE(vcl::scheduler, SCALAR_INT_TYPE)
    VALUE(vcl::scheduler, SCALAR_UINT_TYPE)
    VALUE(vcl::scheduler, SCALAR_LONG_TYPE)
    VALUE(vcl::scheduler, SCALAR_ULONG_TYPE)
    VALUE(vcl::scheduler, SCALAR_HALF_TYPE)
    VALUE(vcl::scheduler, SCALAR_FLOAT_TYPE)
    VALUE(vcl::scheduler, SCALAR_DOUBLE_TYPE)
    
    // vector:
    VALUE(vcl::scheduler, VECTOR_CHAR_TYPE)
    VALUE(vcl::scheduler, VECTOR_UCHAR_TYPE)
    VALUE(vcl::scheduler, VECTOR_SHORT_TYPE)
    VALUE(vcl::scheduler, VECTOR_USHORT_TYPE)
    VALUE(vcl::scheduler, VECTOR_INT_TYPE)
    VALUE(vcl::scheduler, VECTOR_UINT_TYPE)
    VALUE(vcl::scheduler, VECTOR_LONG_TYPE)
    VALUE(vcl::scheduler, VECTOR_ULONG_TYPE)
    VALUE(vcl::scheduler, VECTOR_HALF_TYPE)
    VALUE(vcl::scheduler, VECTOR_FLOAT_TYPE)
    VALUE(vcl::scheduler, VECTOR_DOUBLE_TYPE)
    
    // matrix, row major:
    VALUE(vcl::scheduler, MATRIX_ROW_CHAR_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_UCHAR_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_SHORT_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_USHORT_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_INT_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_UINT_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_LONG_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_ULONG_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_HALF_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_FLOAT_TYPE)
    VALUE(vcl::scheduler, MATRIX_ROW_DOUBLE_TYPE)
    
    // matrix, row major:
    VALUE(vcl::scheduler, MATRIX_COL_CHAR_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_UCHAR_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_SHORT_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_USHORT_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_INT_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_UINT_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_LONG_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_ULONG_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_HALF_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_FLOAT_TYPE)
    VALUE(vcl::scheduler, MATRIX_COL_DOUBLE_TYPE)
    ;

  /*
  typedef vcl::scheduler::statement_node vcl_node_t;

  bp::class_<vcl_node_t>("vcl_statement_node")
    .def_readonly("lhs_type_family", &vcl_node_t::lhs.type_family)
    .def_readonly("lhs_type", &vcl_node_t::lhs.type)
    .def_readonly("rhs_type_family", &vcl_node_t::rhs.type_family)
    .def_readonly("rhs_type", &vcl_node_t::rhs.type)
    .def_readonly("op_family", &vcl_node_t::op.family)
    .def_readonly("op_type", &vcl_node_t::op.type)
    ;
  */

#define STRINGIFY(S) #S
#define SET_OPERAND(I)					\
  .def(STRINGIFY(set_operand_to_ ## I),			\
       &statement_node_wrapper::set_operand_to_ ## I)

  bp::class_<statement_node_wrapper>("statement_node",
				     bp::init<statement_node_wrapper>())
    .def(bp::init<vcl::scheduler::statement_node_type_family,  // lhs
	 vcl::scheduler::statement_node_type,                  // lhs
	 vcl::scheduler::operation_node_type_family,           // op
	 vcl::scheduler::operation_node_type,                  // op
	 vcl::scheduler::statement_node_type_family,           // rhs
	 vcl::scheduler::statement_node_type>())               // rhs
    SET_OPERAND(node_index)
    SET_OPERAND(host_char)
    SET_OPERAND(host_uchar)
    SET_OPERAND(host_short)
    SET_OPERAND(host_ushort)
    SET_OPERAND(host_int)
    SET_OPERAND(host_uint)
    SET_OPERAND(host_long)
    SET_OPERAND(host_ulong)
    SET_OPERAND(host_float)
    SET_OPERAND(host_double)
    SET_OPERAND(scalar_float)
    SET_OPERAND(scalar_double)
    SET_OPERAND(vector_float)
    SET_OPERAND(vector_double)
    SET_OPERAND(matrix_row_float)
    SET_OPERAND(matrix_row_double)
    SET_OPERAND(matrix_col_float)
    SET_OPERAND(matrix_col_double)
    .add_property("vcl_statement_node",
	 bp::make_function(&statement_node_wrapper::get_vcl_statement_node,
			   bp::return_value_policy<bp::return_by_value>()))
    ;

#undef SET_OPERAND

  bp::class_<statement_wrapper>("statement")
    .add_property("size", &statement_wrapper::size)
    .def("execute", &statement_wrapper::execute)
    .def("clear", &statement_wrapper::clear)
    .def("erase_node", &statement_wrapper::erase_node)
    .def("get_node", &statement_wrapper::get_node)
    .def("insert_at_index", &statement_wrapper::insert_at_index)
    .def("insert_at_begin", &statement_wrapper::insert_at_begin)
    .def("insert_at_end", &statement_wrapper::insert_at_end)
    ;
    
}

