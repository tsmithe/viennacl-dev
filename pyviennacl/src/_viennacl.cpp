#include <stdint.h>

#include <iostream>
#include <typeinfo>

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS
#define VIENNACL_WITH_OPENCL

#include <viennacl/backend/memory.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/lanczos.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/power_iter.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/sparse_matrix_operations.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/matrix_proxy.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/vector_proxy.hpp>
#include <viennacl/scheduler/execute.hpp>

namespace vcl = viennacl;
namespace bp = boost::python;
namespace np = boost::numpy;
namespace ublas = boost::numeric::ublas;

typedef void* NoneT;

// TODO: ELIMINATE SPECIFIC DTYPES
typedef vcl::scalar<double> vcl_scalar_t;
typedef double              cpu_scalar_t;

// Sparse types

typedef ublas::compressed_vector<cpu_scalar_t> cpu_sparse_vector_t;

/* Would like to use this, since it's faster than a uBLAS compressed matrix, 
   but it seems to give some sort of type-ambiguity error right now..
typedef ublas::generalized_vector_of_vector<cpu_scalar_t,
					    ublas::row_major,
					    ublas::vector<cpu_sparse_vector_t> >
cpu_gvov_matrix_t;
*/


// TODO: Use ViennaCL operation tags?
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

    /*    
    if (PyObjs & 8) {
      operand1 = static_cast<Operand1T>
	(bp::extract<Operand1T>((bp::api::object)opand1));
    }
    
    if (PyObjs & 4) {
      operand2 = static_cast<Operand2T>
	(bp::extract<Operand2T>((bp::api::object)opand2));
    }
    
    if (PyObjs & 2) {
      operand3 = static_cast<Operand3T>
	(bp::extract<Operand3T>((bp::api::object)opand3));
    }
    
    if (PyObjs & 1) {
      operand4 = static_cast<Operand4T>
	(bp::extract<Operand4T>((bp::api::object)opand4));
    }
    //*/
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

#define OP_TEMPLATE template <class ReturnT,                    \
                              class Operand1T, class Operand2T, \
                              class Operand3T, class Operand4T, \
                              int PyObjs>
#define PYVCL_WORKER_STRUCT(OP) OP_TEMPLATE                \
  struct pyvcl_worker<ReturnT,                             \
                      Operand1T, Operand2T,                \
                      Operand3T, Operand4T,                \
                      OP, PyObjs>
#define DO_OP_FUNC(OP) PYVCL_WORKER_STRUCT(OP) {                     \
  static ReturnT do_op(pyvcl_op<ReturnT,                             \
                       Operand1T, Operand2T,                         \
                       Operand3T, Operand4T,                         \
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
cpu_scalar_t vcl_scalar_to_float(const vcl_scalar_t& vcl_s)
{
  cpu_scalar_t cpu_s = vcl_s;
  return cpu_s;
}

// Vector -- std::vector

template <class SCALARTYPE>
bp::list std_vector_to_list(const std::vector<SCALARTYPE>& v)
{
  bp::list l;
  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((SCALARTYPE)v[i]);
  
  return l;
}

template <class SCALARTYPE>
np::ndarray std_vector_to_ndarray(const std::vector<SCALARTYPE>& v)
{
  return np::from_object(std_vector_to_list<SCALARTYPE>(v),
			 np::dtype::get_builtin<SCALARTYPE>());
}

template <class SCALARTYPE>
boost::shared_ptr<std::vector<SCALARTYPE> >
std_vector_init_ndarray(const np::ndarray& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    bp::throw_error_already_set();
  }
  
  uint32_t s = (uint32_t) array.shape(0);
  
  std::vector<SCALARTYPE> *v = new std::vector<SCALARTYPE>(s);
  
  for (uint32_t i=0; i < s; ++i)
    (*v)[i] = bp::extract<SCALARTYPE>(array[i]);
  
  return boost::shared_ptr<std::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
boost::shared_ptr<std::vector<SCALARTYPE> >
std_vector_init_list(const bp::list& l)
{
  return std_vector_init_ndarray<SCALARTYPE>
    (np::from_object(l, np::dtype::get_builtin<SCALARTYPE>()));
}

template <class SCALARTYPE>
boost::shared_ptr<std::vector<SCALARTYPE> >
std_vector_init_scalar(uint32_t length, SCALARTYPE value) {
  std::vector<SCALARTYPE> *v = new std::vector<SCALARTYPE>(length);
  for (uint32_t i=0; i < length; ++i)
    (*v)[i] = value;
  return boost::shared_ptr<std::vector<SCALARTYPE> >(v);
}

// Vector -- vcl::vector

template <class SCALARTYPE>
bp::list vcl_vector_to_list(const vcl::vector_base<SCALARTYPE>& v)
{
  std::vector<SCALARTYPE> c(v.size());
  vcl::fast_copy(v.begin(), v.end(), c.begin());

  return std_vector_to_list(c);
}

template <class SCALARTYPE>
np::ndarray vcl_vector_to_ndarray(const vcl::vector_base<SCALARTYPE>& v)
{
  return np::from_object(vcl_vector_to_list<SCALARTYPE>(v),
			 np::dtype::get_builtin<SCALARTYPE>());
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_ndarray(const np::ndarray& array)
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
vcl_vector_init_list(const bp::list& l)
{
  return vcl_vector_init_ndarray<SCALARTYPE>
    (np::from_object(l, np::dtype::get_builtin<SCALARTYPE>()));
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_scalar(uint32_t length, SCALARTYPE value)
{
  ublas::scalar_vector<SCALARTYPE> s_v(length, value);
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(length);
  vcl::copy(s_v.begin(), s_v.end(), v->begin());
  return boost::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector_base<SCALARTYPE> >
vcl_range(vcl::vector_base<SCALARTYPE>& vec,
          std::size_t start, std::size_t end)
{
  vcl::range r(start, end);
  vcl::vector_range<vcl::vector_base<SCALARTYPE> > *v_r = new vcl::vector_range
    <vcl::vector_base<SCALARTYPE> >(vec, r);
  return boost::shared_ptr<vcl::vector_base<SCALARTYPE> >(v_r);
}

template <class SCALARTYPE>
boost::shared_ptr<vcl::vector_base<SCALARTYPE> >
vcl_slice(vcl::vector_base<SCALARTYPE>& vec,
          std::size_t start, std::size_t stride, std::size_t size)
{
  vcl::slice slice(start, stride, size);
  vcl::vector_slice<vcl::vector_base<SCALARTYPE> > *v_s = new vcl::vector_slice
    <vcl::vector_base<SCALARTYPE> >(vec, slice);
  return boost::shared_ptr<vcl::vector_base<SCALARTYPE> >(v_s);
}

// Dense matrix

template<class ScalarT>
class ndarray_wrapper
{
  const np::ndarray array; // TODO: Perhaps reference to the wrapped ndarray

public:
  ndarray_wrapper(const np::ndarray& a)
    : array(a)
  { }

  uint32_t size1() const { return array.shape(0); }

  uint32_t size2() const { return array.shape(1); }

  ScalarT operator()(uint32_t row, uint32_t col) const
  {
    return bp::extract<ScalarT>(array[row][col]);
  } 

};

template<class SCALARTYPE, class F>
boost::shared_ptr<vcl::matrix<SCALARTYPE, F> >
matrix_init_scalar(uint32_t n, uint32_t m, SCALARTYPE value)
{
  ublas::scalar_matrix<SCALARTYPE> s_m(n, m, value);
  ublas::matrix<SCALARTYPE> cpu_m(s_m);
  vcl::matrix<SCALARTYPE, F>* mat = new vcl::matrix<SCALARTYPE, F>(n, m);
  vcl::copy(cpu_m, (*mat));
  return boost::shared_ptr<vcl::matrix<SCALARTYPE, F> >(mat);
}


/** @brief Creates the matrix from the supplied ndarray */
template<class SCALARTYPE, class F>
boost::shared_ptr<vcl::matrix<SCALARTYPE, F> >
matrix_init_ndarray(const np::ndarray& array)
{
  int d = array.get_nd();
  if (d != 2) {
    PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
    bp::throw_error_already_set();
  }
  
  ndarray_wrapper<SCALARTYPE> wrapper(array);

  vcl::matrix<SCALARTYPE, F>* mat = new vcl::matrix<SCALARTYPE, F>(wrapper.size1(), wrapper.size2());

  vcl::copy(wrapper, (*mat));
  
  return boost::shared_ptr<vcl::matrix<SCALARTYPE, F> >(mat);
}

template<class SCALARTYPE>
bp::tuple get_strides(const vcl::matrix_base<SCALARTYPE, vcl::row_major>& m) {
  return bp::make_tuple((m.stride1()*m.internal_size2())*sizeof(SCALARTYPE), m.stride2()*sizeof(SCALARTYPE));
}

template<class SCALARTYPE>
bp::tuple get_strides(const vcl::matrix_base<SCALARTYPE, vcl::column_major>& m) {
  return bp::make_tuple(m.stride1()*sizeof(SCALARTYPE), m.stride2()*m.internal_size1()*sizeof(SCALARTYPE));
}

template<class SCALARTYPE>
std::size_t get_offset(const vcl::matrix_base<SCALARTYPE, vcl::row_major>& m) {
  return m.start1()*m.internal_size2() + m.start2();
}

template<class SCALARTYPE>
std::size_t get_offset(const vcl::matrix_base<SCALARTYPE, 
                       vcl::column_major>& m) {
  return m.start1() + m.start2()*m.internal_size1();
}

template<class SCALARTYPE, class VCL_F>
np::ndarray vcl_matrix_to_ndarray(const vcl::matrix_base<SCALARTYPE, VCL_F>& m)
{

  std::size_t size = m.internal_size1() * m.internal_size2() * sizeof(SCALARTYPE);

  SCALARTYPE* data = (SCALARTYPE*)malloc(size);

  vcl::backend::finish();

  // Read the whole matrix
  vcl::backend::memory_read(m.handle(), 0, size, data);
 
  np::dtype dt = np::dtype::get_builtin<SCALARTYPE>();
  bp::tuple shape = bp::make_tuple(m.size1(), m.size2());

  // Delegate determination of strides and start offset to function templates
  bp::tuple strides = get_strides<SCALARTYPE>(m);
  np::ndarray array = np::from_data(data + get_offset<SCALARTYPE>(m),
                                    dt, shape, strides, bp::object());

  //std::cout << "NDARRAY_TEST:\n"
  //        << bp::extract<const char*>(bp::str(array))
  //        << std::endl;

  return array;
}


// Sparse matrix

template <class ScalarType>
class cpu_compressed_matrix_wrapper
{
  // TODO: This is just a quick first implementation. Later, I may well want 
  // TODO: a version that doesn't depend on boost.python types.
  typedef ublas::compressed_matrix<ScalarType, ublas::row_major> ublas_sparse_t;
  ublas_sparse_t cpu_compressed_matrix;

public:
  bp::list places;

  cpu_compressed_matrix_wrapper()
  {
    cpu_compressed_matrix = ublas_sparse_t(0,0,0);
  }

  cpu_compressed_matrix_wrapper(uint32_t _size1, uint32_t _size2)
  {
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2);
  }

  cpu_compressed_matrix_wrapper(uint32_t _size1, uint32_t _size2, uint32_t _nnz)
  {
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2, _nnz);
  }

  cpu_compressed_matrix_wrapper(const cpu_compressed_matrix_wrapper& w)
    : cpu_compressed_matrix(w.cpu_compressed_matrix)
  {
    update_places();
  }

  template<class SparseT>
  cpu_compressed_matrix_wrapper(const SparseT& vcl_sparse_matrix)
  {
    cpu_compressed_matrix = ublas_sparse_t(vcl_sparse_matrix.size1(),
                                           vcl_sparse_matrix.size2());
    vcl::copy(vcl_sparse_matrix, cpu_compressed_matrix);
    
    update_places();
  }

  cpu_compressed_matrix_wrapper(const np::ndarray& array)
  {

    int d = array.get_nd();
    if (d != 2) {
      PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
      bp::throw_error_already_set();
    }
    
    uint32_t n = array.shape(0);
    uint32_t m = array.shape(1);
    
    cpu_compressed_matrix = ublas_sparse_t(n, m);
    
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < m; ++j) {
	ScalarType val = bp::extract<ScalarType>(array[i][j]);
	if (val != 0) {
	  cpu_compressed_matrix(i, j) = val;
	  places.append(bp::make_tuple(i, j));
	}
      }
    }
    
  }

  np::ndarray as_ndarray()
  {

    np::dtype dt = np::dtype::get_builtin<ScalarType>();
    bp::tuple shape = bp::make_tuple(size1(), size2());
    
    np::ndarray array = np::zeros(shape, dt);
  
    for (std::size_t i = 0; i < bp::len(places); ++i) {
      bp::tuple coord = bp::extract<bp::tuple>(places[i]);
      uint32_t x = bp::extract<uint32_t>(coord[0]);
      uint32_t y = bp::extract<uint32_t>(coord[1]);
      array[x][y] = get(x, y);
    }

    return array;

  }

  template<class SparseT>
  boost::shared_ptr<SparseT>
  as_vcl_sparse_matrix()
  {
    SparseT* vcl_sparse_matrix = new SparseT();
    vcl::copy(cpu_compressed_matrix, *vcl_sparse_matrix);
    return boost::shared_ptr<SparseT>(vcl_sparse_matrix);
  }

  template<class SparseT>
  boost::shared_ptr<SparseT>
  as_vcl_sparse_matrix_with_size()
  {
    SparseT* vcl_sparse_matrix = new SparseT(size1(), size2(), nnz());
    vcl::copy(cpu_compressed_matrix, *vcl_sparse_matrix);
    return boost::shared_ptr<SparseT>(vcl_sparse_matrix);
  }

  void update_places()
  {
    typedef typename ublas_sparse_t::iterator1 it1;
    typedef typename ublas_sparse_t::iterator2 it2;

    for (it1 i = cpu_compressed_matrix.begin1();
         i != cpu_compressed_matrix.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {

	if (cpu_compressed_matrix(j.index1(), j.index2()) != 0) {
          bp::tuple coord = bp::make_tuple(j.index1(), j.index2());
          if (not places.count(coord)) {
            places.append(coord);
          } else {
            while (places.count(coord) > 1) {
              places.remove(coord);
            }
          }          
        }

      }
    }

    nnz();

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
      if (cpu_compressed_matrix(n, m) == 0)
	places.remove(item);
      else
	++i;

    } 
      
    return bp::len(places);

  }

  uint32_t size1() const
  {
    return cpu_compressed_matrix.size1();
  }

  uint32_t size2() const
  {
    return cpu_compressed_matrix.size2();
  }

  void resize(uint32_t _size1, uint32_t _size2)
  {
  
    if (_size1 < size1())
      _size1 = size1();

    if (_size2 < size2())
      _size2 = size2();

    if ((_size1 == size1()) and (_size2 == size2()))
      return;

    // TODO NB: ublas compressed matrix does not support preserve on resize
    //          so this below is annoyingly hacky...

    cpu_compressed_matrix_wrapper temp(*this);
    cpu_compressed_matrix.resize(_size1, _size2, false); // preserve == false!

    for (std::size_t i = 0; i < bp::len(places); ++i) {
      bp::tuple coord = bp::extract<bp::tuple>(places[i]);
      uint32_t x = bp::extract<uint32_t>(coord[0]);
      uint32_t y = bp::extract<uint32_t>(coord[1]);
      cpu_compressed_matrix(x, y) = temp.get(x, y);
    }

  }

  void set(uint32_t n, uint32_t m, ScalarType val) 
  {
    if (n >= size1() or m >= size2())
      resize(n+1, m+1);

    // We want to keep track of which places are filled.
    bp::tuple coord = bp::make_tuple(n, m);
    if (not places.count(coord))
      places.append(coord);

    cpu_compressed_matrix(n, m) = val;
  }

  // Need this because bp cannot deal with operator()
  ScalarType get(uint32_t n, uint32_t m)
  {
    return cpu_compressed_matrix(n, m);
  }

};

/************************************************
        Scheduler / generator interface  
 ************************************************/

class statement_node_wrapper {

  vcl::scheduler::statement::value_type vcl_node;

public:

  statement_node_wrapper(const statement_node_wrapper& node)
    : vcl_node(node.vcl_node)
  { }

  statement_node_wrapper(vcl::scheduler::statement_node node)
    : vcl_node(node)
  { }

  statement_node_wrapper(vcl::scheduler::statement_node_type_family lhs_family,
			 vcl::scheduler::statement_node_subtype lhs_subtype,
			 vcl::scheduler::statement_node_numeric_type lhs_numeric_type,
			 vcl::scheduler::operation_node_type_family op_family,
			 vcl::scheduler::operation_node_type op_type,
			 vcl::scheduler::statement_node_type_family rhs_family,
			 vcl::scheduler::statement_node_subtype rhs_subtype,
			 vcl::scheduler::statement_node_numeric_type rhs_numeric_type)
  {
    vcl_node.op.type_family = op_family;
    vcl_node.op.type = op_type;
    vcl_node.lhs.type_family = lhs_family;
    vcl_node.lhs.subtype = lhs_subtype;
    vcl_node.lhs.numeric_type = lhs_numeric_type;
    vcl_node.rhs.type_family = rhs_family;
    vcl_node.rhs.subtype = rhs_subtype;
    vcl_node.rhs.numeric_type = rhs_numeric_type;
  }

  vcl::scheduler::statement_node& get_vcl_statement_node()
  {
    return vcl_node;
  }

  vcl::scheduler::statement_node get_vcl_statement_node() const
  {
    return vcl_node;
  }

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
      throw vcl::scheduler::statement_not_supported_exception      \
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
  SET_OPERAND(vcl::vector_base<float>*, vector_float)
  SET_OPERAND(vcl::vector_base<double>*, vector_double)

  SET_OPERAND(vcl::implicit_vector_base<float>*, implicit_vector_float)
  SET_OPERAND(vcl::implicit_vector_base<double>*, implicit_vector_double)

  // NB: need to add remaining matrix_row types as they become available
  SET_OPERAND(CONCAT(vcl::matrix_base<float, vcl::row_major>*),
              matrix_row_float)
  SET_OPERAND(CONCAT(vcl::matrix_base<double, vcl::row_major>*),
              matrix_row_double)
  
  // NB: need to add remaining matrix_col types as they become available
  SET_OPERAND(CONCAT(vcl::matrix_base<float, vcl::column_major>*),
              matrix_col_float)
  SET_OPERAND(CONCAT(vcl::matrix_base<double, vcl::column_major>*),
              matrix_col_double)

  SET_OPERAND(vcl::implicit_matrix_base<float>*, implicit_matrix_float)
  SET_OPERAND(vcl::implicit_matrix_base<double>*, implicit_matrix_double)

  SET_OPERAND(vcl::compressed_matrix<float>*, compressed_matrix_float)
  SET_OPERAND(vcl::compressed_matrix<double>*, compressed_matrix_double)

  SET_OPERAND(vcl::coordinate_matrix<float>*, coordinate_matrix_float)
  SET_OPERAND(vcl::coordinate_matrix<double>*, coordinate_matrix_double)

  SET_OPERAND(vcl::ell_matrix<float>*, ell_matrix_float)
  SET_OPERAND(vcl::ell_matrix<double>*, ell_matrix_double)

  SET_OPERAND(vcl::hyb_matrix<float>*, hyb_matrix_float)
  SET_OPERAND(vcl::hyb_matrix<double>*, hyb_matrix_double)

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
   Generic std::vector wrapper
 *******************************/

//
// Do I really need this?...
//

template <class T>
class vector_wrapper {
  std::vector<T> vector_;
  typedef typename std::vector<T>::iterator iter_t;
  typedef typename std::vector<T>::const_iterator const_iter_t;

public:
  vector_wrapper(bp::list list_) {
    for (std::size_t n = 0; n < bp::len(list_); ++n)
      vector_.push_back(bp::extract<T>(list_[n]));
  }

  vector_wrapper(std::vector<T> v) : vector_(v) { }

  bp::list to_list() {
    bp::list l;
    for (iter_t it = vector_.begin(); it != vector_.end(); it++)
      l.append(*it);
    return l;
  }

  std::size_t size() const {
    return vector_.size();
  }

  void clear() {
    vector_.clear();
  }

  std::vector<T> get() const {
    return vector_;
  }
  
  T get(std::size_t offset) const {
    return vector_[offset];
  }

  void erase(std::size_t offset)
  {
    iter_t it = vector_.begin();
    vector_.erase(it+offset);
  }

  void insert_at_index(std::size_t offset, const T& item)
  {
    iter_t it = vector_.begin();
    vector_.insert(it+offset, item);
  }

  void insert_at_begin(const T& item)
  {
    iter_t it = vector_.begin();
    vector_.insert(it, item);
  }

  void insert_at_end(const T& item)
  {
    vector_.push_back(item);
  }

};

//////////////////////////////////// ...

template <class T>
std::vector<T> vector_from_list(bp::list l) {
  std::vector<T> v(bp::len(l));
  for (std::size_t n = 0; n < bp::len(l); n++)
    v.push_back(bp::extract<T>(l[n]));
  return v;
}

template <class T>
bp::list list_from_vector(std::vector<T> v) {
  bp::list l;
  for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); it++)
    l.append(*it);
  return l;
}

/*******************************
  Python module initialisation
 *******************************/

#define DISAMBIGUATE_FUNCTION_PTR(RET, OLD_NAME, NEW_NAME, ARGS) \
  RET (*NEW_NAME) ARGS = &OLD_NAME;

#define DISAMBIGUATE_CLASS_FUNCTION_PTR(CLASS, RET, OLD_NAME, NEW_NAME, ARGS)\
  RET (CLASS::*NEW_NAME) ARGS = &CLASS::OLD_NAME;

void translate_string_exception(const char* e)
{
  // Use the Python 'C' API to set up an exception object
  PyErr_SetString(PyExc_RuntimeError, e);
}


BOOST_PYTHON_MODULE(_viennacl)
{

  bp::register_exception_translator
    <const char*>
    (&translate_string_exception);

  np::initialize();

  // --------------------------------------------------

  // *** Utility functions ***
  bp::def("backend_finish", vcl::backend::finish);

  // --------------------------------------------------

  // *** Scalar type ***

  // TODO: EXPOSE ALL NUMERIC TYPES

  bp::class_<vcl_scalar_t>("scalar_double")
    // Utility functions
    .def(bp::init<float>())
    .def(bp::init<int>())
    .def("as_double", &vcl_scalar_to_float)

    /*
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
    * /

    // Scalar-matrix operations
    .def("__mul__", pyvcl_do_2ary_op<vcl::matrix<float, vcl::row_major>,
	 vcl_scalar_t&, vcl::matrix<float, vcl::row_major>&,
	 op_mul, 0>)
    .def("__mul__", pyvcl_do_2ary_op<vcl::matrix<float, vcl::row_major>,
	 vcl_scalar_t&, vcl::matrix<float, vcl::row_major>&,
	 op_mul, 0>)
    .def("__mul__", pyvcl_do_2ary_op<vcl::matrix<double, vcl::column_major>,
	 vcl_scalar_t&, vcl::matrix<double, vcl::column_major>&,
	 op_mul, 0>)
    .def("__mul__", pyvcl_do_2ary_op<vcl::matrix<double, vcl::column_major>,
	 vcl_scalar_t&, vcl::matrix<double, vcl::column_major>&,
	 op_mul, 0>)
    */
    ;

  // --------------------------------------------------

  bp::class_<vcl::range>("range",
                         bp::init<std::size_t, std::size_t>());
  bp::class_<vcl::slice>("slice",
                         bp::init<std::size_t, std::size_t, std::size_t>());

  // DISAMBIGUATE_FUNCTION_PTR(RET, OLD_NAME, NEW_NAME, ARGS)  

  // *** Vector types ***

#define EXPORT_VECTOR_CLASS(TYPE, NAME)					\
  bp::class_<vcl::vector_base<TYPE>,                                    \
	     boost::shared_ptr<vcl::vector_base<TYPE> > >               \
    ("vector_base", bp::no_init)                                        \
    .def("as_ndarray", &vcl_vector_to_ndarray<TYPE>)			\
    .def("as_list", &vcl_vector_to_list<TYPE>)                          \
    .def("clear", &vcl::vector_base<TYPE>::clear)                       \
    .add_property("size", &vcl::vector_base<TYPE>::size)                \
    .add_property("internal_size", &vcl::vector_base<TYPE>::internal_size) \
    .add_property("index_norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,	\
		  vcl::vector_base<TYPE>&,                              \
		  op_index_norm_inf, 0>)				\
    .def("__add__", pyvcl_do_2ary_op<vcl::vector<TYPE>,                 \
	 vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,              \
	 op_add, 0>)							\
    .def("__iadd__", pyvcl_do_2ary_op<vcl::vector<TYPE>,                \
	 vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,              \
	 op_iadd, 0>)							\
    .def("__mul__", pyvcl_do_2ary_op<vcl::vector<TYPE>,                 \
	 vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,              \
	 op_element_prod, 0>)                                           \
    .def("outer", pyvcl_do_2ary_op<vcl::matrix<TYPE, vcl::row_major>,   \
	 vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,              \
	 op_outer_prod, 0>)                                             \
    .def("outer", pyvcl_do_2ary_op<vcl::matrix<TYPE, vcl::column_major>, \
	 vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,              \
	 op_outer_prod, 0>)                                             \
    ;                                                                   \
  bp::class_<vcl::vector_range<vcl::vector_base<TYPE> >,                \
             boost::shared_ptr<vcl::vector_range<vcl::vector_base<TYPE> > >, \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ("vector_range", bp::no_init);                                      \
  bp::class_<vcl::vector_slice<vcl::vector_base<TYPE> >,                \
             boost::shared_ptr<vcl::vector_slice<vcl::vector_base<TYPE> > >, \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ("vector_slice", bp::no_init);                                      \
  bp::class_<vcl::vector<TYPE>,						\
	     boost::shared_ptr<vcl::vector<TYPE> >,                     \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ( NAME )								\
    .def(bp::init<int>())						\
    .def(bp::init<vcl::vector<TYPE> >())				\
    .def("__init__", bp::make_constructor(vcl_vector_init_ndarray<TYPE>)) \
    .def("__init__", bp::make_constructor(vcl_vector_init_list<TYPE>))	\
    .def("__init__", bp::make_constructor(vcl_vector_init_scalar<TYPE>))\
    ;                                                                   \
  bp::def("plane_rotation", pyvcl_do_4ary_op<bp::object,                \
	  vcl::vector<TYPE>&, vcl::vector<TYPE>&,                       \
	  TYPE, TYPE,                                                   \
	  op_plane_rotation, 0>);                                       \
  bp::class_<std::vector<TYPE>,						\
	     boost::shared_ptr<std::vector<TYPE> > >			\
    ( "std_" NAME )                                                     \
    .def(bp::init<int>())						\
    .def(bp::init<std::vector<TYPE> >())				\
    .def("__init__", bp::make_constructor(std_vector_init_ndarray<TYPE>)) \
    .def("__init__", bp::make_constructor(std_vector_init_list<TYPE>))	\
    .def("__init__", bp::make_constructor(std_vector_init_scalar<TYPE>))\
    .def("as_ndarray", &std_vector_to_ndarray<TYPE>)                    \
    .def("as_list", &std_vector_to_list<TYPE>)                          \
    .add_property("size", &std::vector<TYPE>::size)			\
    ;                                                                   \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_range<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_##TYPE##_range,              \
                            (CONCAT(vcl::vector_base<TYPE>&,            \
                                    const vcl::range&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_range<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_range_##TYPE##_range,        \
                            (CONCAT(vcl::vector_range<                  \
                                    vcl::vector_base<TYPE> >&,          \
                                    const vcl::range&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_slice<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_##TYPE##_slice,              \
                            (CONCAT(vcl::vector_base<TYPE>&,            \
                                    const vcl::slice&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_slice<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_slice_##TYPE##_slice,        \
                            (CONCAT(vcl::vector_slice<                  \
                                    vcl::vector_base<TYPE> >&,          \
                                    const vcl::slice&)))                \
  bp::def("project", project_vector_##TYPE##_range);                    \
  bp::def("project", project_vector_range_##TYPE##_range);              \
  bp::def("project", project_vector_##TYPE##_slice);                    \
  bp::def("project", project_vector_slice_##TYPE##_slice);
  // TODO: range/slice interactions

  EXPORT_VECTOR_CLASS(float, "vector_float")
  EXPORT_VECTOR_CLASS(double, "vector_double")


  // --------------------------------------------------
  
  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, unsigned int,
                                  iters, get_cg_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, double,
                                  error, get_cg_error, () const)
  bp::class_<vcl::linalg::cg_tag>("cg_tag")
    .def(bp::init<double, unsigned int>())
    .add_property("tolerance", &vcl::linalg::cg_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::cg_tag::max_iterations)
    .add_property("iters", get_cg_iters)
    .add_property("error", get_cg_error)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, std::size_t,
                                  iters, get_bicgstab_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, double,
                                  error, get_bicgstab_error, () const)
  bp::class_<vcl::linalg::bicgstab_tag>("bicgstab_tag")
    .def(bp::init<double, std::size_t, std::size_t>())
    .add_property("tolerance", &vcl::linalg::bicgstab_tag::tolerance)
    .add_property("max_iterations",
                  &vcl::linalg::bicgstab_tag::max_iterations)
    .add_property("max_iterations_before_restart",
                  &vcl::linalg::bicgstab_tag::max_iterations_before_restart)
    .add_property("iters", get_bicgstab_iters)
    .add_property("error", get_bicgstab_error)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, unsigned int,
                                  iters, get_gmres_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, double,
                                  error, get_gmres_error, () const)
  bp::class_<vcl::linalg::gmres_tag>("gmres_tag")
    .def(bp::init<double, unsigned int, unsigned int>())
    .add_property("tolerance", &vcl::linalg::gmres_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::gmres_tag::max_iterations)
    .add_property("iters", get_gmres_iters)
    .add_property("error", get_gmres_error)
    .add_property("krylov_dim", &vcl::linalg::gmres_tag::krylov_dim)
    .add_property("max_restarts", &vcl::linalg::gmres_tag::max_restarts)
    ;

  // *** Dense matrix type ***

#define EXPORT_DENSE_MATRIX_CLASS(TYPE, LAYOUT, F, CPU_F)               \
  bp::class_<vcl::matrix_base<TYPE, F>,                                 \
	     boost::shared_ptr<vcl::matrix_base<TYPE, F> > >            \
    ("matrix_base", bp::no_init)                                        \
    .def("as_ndarray", &vcl_matrix_to_ndarray<TYPE, F>)                 \
    .def("clear", &vcl::matrix_base<TYPE, F>::clear)                    \
    .add_property("size1", &vcl::matrix_base<TYPE, F>::size1)           \
    .add_property("internal_size1",                                     \
                  &vcl::matrix_base<TYPE, F>::internal_size1)           \
    .add_property("size2", &vcl::matrix_base<TYPE, F>::size2)           \
    .add_property("internal_size2",                                     \
                  &vcl::matrix_base<TYPE, F>::internal_size2)           \
    .add_property("trans", pyvcl_do_1ary_op<vcl::matrix<TYPE, F>,       \
                  vcl::matrix_base<TYPE, F>&,                           \
                  op_trans, 0>)                                         \
    .def("__add__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,              \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     op_add, 0>)                        \
    .def("__sub__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,              \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     op_sub, 0>)                        \
    .def("__mul__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,              \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     vcl_scalar_t&,                     \
                                     op_mul, 0>)                        \
    .def("__mul__", pyvcl_do_2ary_op<vcl::vector<TYPE>,                 \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     vcl::vector_base<TYPE>&,           \
                                     op_prod, 0>)                       \
    .def("__mul__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,              \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     vcl::matrix_base<TYPE, F>&,        \
                                     op_prod, 0>)                       \
    .def("__truediv__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,          \
                                         vcl::matrix_base<TYPE, F>&,    \
                                         vcl_scalar_t&,                 \
                                         op_div, 0>)                    \
    .def("__iadd__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,             \
                                      vcl::matrix_base<TYPE, F>&,       \
                                      vcl::matrix_base<TYPE, F>&,       \
                                      op_iadd, 0>)                      \
    .def("__isub__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,             \
                                      vcl::matrix_base<TYPE, F>&,       \
                                      vcl::matrix_base<TYPE, F>&,       \
                                      op_isub, 0>)                      \
    .def("__imul__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,             \
                                      vcl::matrix_base<TYPE, F>&,       \
                                      vcl_scalar_t&,                    \
                                      op_imul, 0>)                      \
    .def("__itruediv__", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,         \
                                          vcl::matrix_base<TYPE, F>&,   \
                                          vcl_scalar_t&,                \
                                          op_idiv, 0>)                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,           \
	 vcl::linalg::lower_tag&,                                       \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,           \
	 vcl::linalg::unit_lower_tag&,                                  \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,           \
	 vcl::linalg::upper_tag&,                                       \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,           \
	 vcl::linalg::unit_upper_tag&,                                  \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix<TYPE, F>&, vcl::vector<TYPE>&,                     \
	 vcl::linalg::cg_tag&,                                          \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix<TYPE, F>&, vcl::vector<TYPE>&,                     \
	 vcl::linalg::bicgstab_tag&,                                    \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,                   \
	 vcl::matrix<TYPE, F>&, vcl::vector<TYPE>&,                     \
	 vcl::linalg::gmres_tag&,                                       \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>,                \
	 vcl::matrix_base<TYPE, F>&, vcl::matrix_base<TYPE, F>&,        \
	 vcl::linalg::lower_tag&,                                       \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>,                \
	 vcl::matrix_base<TYPE, F>&, vcl::matrix_base<TYPE, F>&,        \
	 vcl::linalg::unit_lower_tag&,                                  \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>,                \
	 vcl::matrix_base<TYPE, F>&, vcl::matrix_base<TYPE, F>&,        \
	 vcl::linalg::upper_tag&,                                       \
	 op_solve, 0>)                                                  \
    .def("solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>,                \
	 vcl::matrix_base<TYPE, F>&, vcl::matrix_base<TYPE, F>&,        \
	 vcl::linalg::unit_upper_tag&,                                  \
	 op_solve, 0>)                                                  \
    ;                                                                   \
  bp::class_<vcl::matrix_range<vcl::matrix_base<TYPE, F> >,             \
             boost::shared_ptr<vcl::matrix_range<vcl::matrix_base<TYPE, \
                                                                  F> > >, \
             bp::bases<vcl::matrix_base<TYPE, F> > >                    \
    ("matrix_range", bp::no_init);                                      \
  bp::class_<vcl::matrix_slice<vcl::matrix_base<TYPE, F> >,             \
             boost::shared_ptr<vcl::matrix_slice<vcl::matrix_base<TYPE, \
                                                                  F> > >, \
             bp::bases<vcl::matrix_base<TYPE, F> > >                    \
    ("matrix_slice", bp::no_init);                                      \
  bp::class_<vcl::matrix<TYPE, F>,                                      \
             boost::shared_ptr<vcl::matrix<TYPE, F> >,                  \
             bp::bases<vcl::matrix_base<TYPE, F> > >                    \
    ( "matrix_" #LAYOUT "_" #TYPE )                                     \
    .def(bp::init<vcl::matrix<TYPE, F> >())                             \
    .def(bp::init<uint32_t, uint32_t>())                                \
    .def("__init__", bp::make_constructor(matrix_init_ndarray<TYPE, F>))\
    .def("__init__", bp::make_constructor(matrix_init_scalar<TYPE, F>)) \
    ;                                                                   \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_range<                   \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_##TYPE##_##LAYOUT##_range_range, \
                            (CONCAT(vcl::matrix_base<TYPE, F>&,         \
                                    const vcl::range&,                  \
                                    const vcl::range&)))                \
 DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_range<                    \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_range_##TYPE##_##LAYOUT##_range_range, \
                            (CONCAT(vcl::matrix_range<                  \
                                    vcl::matrix_base<TYPE, F> >&,       \
                                    const vcl::range&,                  \
                                    const vcl::range&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_slice<                   \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_##TYPE##_##LAYOUT##_slice_slice, \
                            (CONCAT(vcl::matrix_base<TYPE, F>&,         \
                                    const vcl::slice&,                  \
                                    const vcl::slice&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_slice<                   \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_slice_##TYPE##_##LAYOUT##_slice_slice, \
                            (CONCAT(vcl::matrix_slice<                  \
                                    vcl::matrix_base<TYPE, F> >&,       \
                                    const vcl::slice&,                  \
                                    const vcl::slice&)))                \
  bp::def("project", project_matrix_##TYPE##_##LAYOUT##_range_range);   \
  bp::def("project", project_matrix_range_##TYPE##_##LAYOUT##_range_range); \
  bp::def("project", project_matrix_##TYPE##_##LAYOUT##_slice_slice);   \
  bp::def("project", project_matrix_slice_##TYPE##_##LAYOUT##_slice_slice);
  // TODO: range/slice interactions

  EXPORT_DENSE_MATRIX_CLASS(double, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(float, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(double, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(float, col, vcl::column_major, ublas::column_major)

           
  // --------------------------------------------------

  // *** Sparse matrix types ***

    bp::class_<vcl::compressed_matrix<double>,
               boost::shared_ptr<vcl::compressed_matrix<double> > >
    ("compressed_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::compressed_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::compressed_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("nnz",
		  make_function(&vcl::compressed_matrix<double>::nnz,
			      bp::return_value_policy<bp::return_by_value>()))

    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)
    
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve, 0>)
    ;

  bp::class_<vcl::coordinate_matrix<double>, 
             boost::shared_ptr<vcl::coordinate_matrix<double> >,
             boost::noncopyable >
    ("coordinate_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::coordinate_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::coordinate_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("nnz",
		  make_function(&vcl::coordinate_matrix<double>::nnz,
			      bp::return_value_policy<bp::return_by_value>()))

    //*
    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::coordinate_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)

    /* 
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<double>,
	 vcl::coordinate_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<double>,
	 vcl::coordinate_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<double>,
	 vcl::coordinate_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<double>,
	 vcl::coordinate_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve, 0>)*/
    ;

  bp::class_<vcl::ell_matrix<double>, 
             boost::shared_ptr<vcl::ell_matrix<double> >,
             boost::noncopyable >
    ("ell_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::ell_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::ell_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("nnz",
		  make_function(&vcl::ell_matrix<double>::nnz,
			      bp::return_value_policy<bp::return_by_value>()))

    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::ell_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)
    ;

  bp::class_<vcl::hyb_matrix<double>, 
             boost::shared_ptr<vcl::hyb_matrix<double> >,
             boost::noncopyable >
    ("hyb_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::hyb_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::hyb_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::hyb_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)
    ;

  bp::class_<cpu_compressed_matrix_wrapper<double> >
    ("cpu_compressed_matrix_double")
    .def(bp::init<>())
    .def(bp::init<uint32_t, uint32_t>())
    .def(bp::init<uint32_t, uint32_t, uint32_t>())
    .def(bp::init<cpu_compressed_matrix_wrapper<double> >())
    .def(bp::init<vcl::compressed_matrix<double> >())
    //.def(bp::init<vcl::coordinate_matrix<double> >())
    .def(bp::init<vcl::ell_matrix<double> >())
    .def(bp::init<vcl::hyb_matrix<double> >())
    .def(bp::init<np::ndarray>())
    .def_readonly("nonzeros", &cpu_compressed_matrix_wrapper<double>::places)
    .add_property("nnz", &cpu_compressed_matrix_wrapper<double>::nnz)
    .add_property("size1", &cpu_compressed_matrix_wrapper<double>::size1)
    .add_property("size2", &cpu_compressed_matrix_wrapper<double>::size2)
    .def("resize", &cpu_compressed_matrix_wrapper<double>::resize)
    .def("set", &cpu_compressed_matrix_wrapper<double>::set)
    .def("get", &cpu_compressed_matrix_wrapper<double>::get)
    .def("as_ndarray", &cpu_compressed_matrix_wrapper<double>::as_ndarray)
    .def("as_compressed_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix_with_size<vcl::compressed_matrix<double> >)
    .def("as_coordinate_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix_with_size<vcl::coordinate_matrix<double> >)
    .def("as_ell_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix<vcl::ell_matrix<double> >)
    .def("as_hyb_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix<vcl::hyb_matrix<double> >)
    ;

  // --------------------------------------------------

  // Eigenvalue computations

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::power_iter_tag, double,
                                  factor, get_power_iter_factor, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::power_iter_tag, std::size_t,
                                  max_iterations,
                                  get_power_iter_max_iterations, () const)
  bp::class_<vcl::linalg::power_iter_tag>("power_iter_tag")
    .def(bp::init<double, std::size_t>())
    .add_property("factor", get_power_iter_factor)
    .add_property("max_iterations", get_power_iter_max_iterations)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, std::size_t,
                                  num_eigenvalues,
                                  get_lanczos_num_eigenvalues, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, double,
                                  factor, get_lanczos_factor, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, std::size_t,
                                  krylov_size, get_lanczos_krylov_size,
                                  () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, int,
                                  method, get_lanczos_method, () const)
  bp::class_<vcl::linalg::lanczos_tag>("lanczos_tag")
    .def(bp::init<double, std::size_t, int, std::size_t>())
    .add_property("num_eigenvalues", get_lanczos_num_eigenvalues)
    .add_property("factor", get_lanczos_factor)
    .add_property("krylov_size", get_lanczos_krylov_size)
    .add_property("method", get_lanczos_method)
    ;

  DISAMBIGUATE_FUNCTION_PTR(double, 
                            vcl::linalg::eig,eig_power_iter_double_row,
                            (const vcl::matrix<double, vcl::row_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_row);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_row,
                            (const vcl::matrix<double, vcl::row_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_row);

  DISAMBIGUATE_FUNCTION_PTR(float, 
                            vcl::linalg::eig,eig_power_iter_float_row,
                            (const vcl::matrix<float, vcl::row_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_row);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_row,
                            (const vcl::matrix<float, vcl::row_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_row);

  DISAMBIGUATE_FUNCTION_PTR(double, 
                            vcl::linalg::eig,eig_power_iter_double_col,
                            (const vcl::matrix<double, vcl::column_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_col);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_col,
                            (const vcl::matrix<double, vcl::column_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_col);

  DISAMBIGUATE_FUNCTION_PTR(float, 
                            vcl::linalg::eig,eig_power_iter_float_col,
                            (const vcl::matrix<float, vcl::column_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_col);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_col,
                            (const vcl::matrix<float, vcl::column_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_col);

  // --------------------------------------------------

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
    VALUE(vcl::scheduler, OPERATION_UNARY_TRANS_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_NORM_1_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_NORM_2_TYPE)
    VALUE(vcl::scheduler, OPERATION_UNARY_NORM_INF_TYPE)
    
    // binary expression
    VALUE(vcl::scheduler, OPERATION_BINARY_ACCESS_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_ASSIGN_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_INPLACE_ADD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_INPLACE_SUB_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_ADD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_SUB_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_MAT_VEC_PROD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_MAT_MAT_PROD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_MULT_TYPE)// scalar * vector/matrix
    VALUE(vcl::scheduler, OPERATION_BINARY_DIV_TYPE) // vector/matrix / scalar
    VALUE(vcl::scheduler, OPERATION_BINARY_ELEMENT_PROD_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_ELEMENT_DIV_TYPE)
    VALUE(vcl::scheduler, OPERATION_BINARY_INNER_PROD_TYPE)
    ;

  bp::enum_<vcl::scheduler::statement_node_type_family>
    ("statement_node_type_family")
    VALUE(vcl::scheduler, COMPOSITE_OPERATION_FAMILY)
    VALUE(vcl::scheduler, SCALAR_TYPE_FAMILY)
    VALUE(vcl::scheduler, VECTOR_TYPE_FAMILY)
    VALUE(vcl::scheduler, MATRIX_TYPE_FAMILY)
    ;

  bp::enum_<vcl::scheduler::statement_node_subtype>
    ("statement_node_subtype")
    VALUE(vcl::scheduler, INVALID_SUBTYPE)

    VALUE(vcl::scheduler, HOST_SCALAR_TYPE)
    VALUE(vcl::scheduler, DEVICE_SCALAR_TYPE)

    VALUE(vcl::scheduler, DENSE_VECTOR_TYPE)
    VALUE(vcl::scheduler, IMPLICIT_VECTOR_TYPE)

    VALUE(vcl::scheduler, DENSE_ROW_MATRIX_TYPE)
    VALUE(vcl::scheduler, DENSE_COL_MATRIX_TYPE)
    VALUE(vcl::scheduler, IMPLICIT_MATRIX_TYPE)

    VALUE(vcl::scheduler, COMPRESSED_MATRIX_TYPE)
    VALUE(vcl::scheduler, COORDINATE_MATRIX_TYPE)
    VALUE(vcl::scheduler, ELL_MATRIX_TYPE)
    VALUE(vcl::scheduler, HYB_MATRIX_TYPE)
    ;

  bp::enum_<vcl::scheduler::statement_node_numeric_type>
    ("statement_node_numeric_type")
    VALUE(vcl::scheduler, INVALID_NUMERIC_TYPE)

    VALUE(vcl::scheduler, CHAR_TYPE)
    VALUE(vcl::scheduler, UCHAR_TYPE)
    VALUE(vcl::scheduler, SHORT_TYPE)
    VALUE(vcl::scheduler, USHORT_TYPE)
    VALUE(vcl::scheduler, INT_TYPE)
    VALUE(vcl::scheduler, UINT_TYPE)
    VALUE(vcl::scheduler, LONG_TYPE)
    VALUE(vcl::scheduler, ULONG_TYPE)
    VALUE(vcl::scheduler, HALF_TYPE)
    VALUE(vcl::scheduler, FLOAT_TYPE)
    VALUE(vcl::scheduler, DOUBLE_TYPE)
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

DISAMBIGUATE_CLASS_FUNCTION_PTR(statement_node_wrapper,         // class
                                vcl::scheduler::statement_node, // ret. type
                                get_vcl_statement_node,         // old_name
                                get_vcl_statement_node,         // new_name
                                () const)                       // args

  bp::class_<statement_node_wrapper>("statement_node",
				     bp::init<statement_node_wrapper>())
    .def(bp::init<vcl::scheduler::statement_node_type_family,  // lhs
	 vcl::scheduler::statement_node_subtype,               // lhs
	 vcl::scheduler::statement_node_numeric_type,          // lhs
	 vcl::scheduler::operation_node_type_family,           // op
	 vcl::scheduler::operation_node_type,                  // op
	 vcl::scheduler::statement_node_type_family,           // rhs
	 vcl::scheduler::statement_node_subtype,               // rhs
	 vcl::scheduler::statement_node_numeric_type>())       // rhs
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

  // TODO: DIFFERENT DTYPES

    SET_OPERAND(scalar_float)
    SET_OPERAND(scalar_double)
    SET_OPERAND(vector_float)
    SET_OPERAND(vector_double)
    SET_OPERAND(implicit_vector_float)
    SET_OPERAND(implicit_vector_double)
    SET_OPERAND(matrix_row_float)
    SET_OPERAND(matrix_row_double)
    SET_OPERAND(matrix_col_float)
    SET_OPERAND(matrix_col_double)
    SET_OPERAND(implicit_matrix_float)
    SET_OPERAND(implicit_matrix_double)
    SET_OPERAND(implicit_vector_float)
    SET_OPERAND(implicit_vector_double)
    SET_OPERAND(compressed_matrix_float)
    SET_OPERAND(compressed_matrix_double)
    SET_OPERAND(coordinate_matrix_float)
    SET_OPERAND(coordinate_matrix_double)
    SET_OPERAND(ell_matrix_float)
    SET_OPERAND(ell_matrix_double)
    SET_OPERAND(hyb_matrix_float)
    SET_OPERAND(hyb_matrix_double)

    .add_property("vcl_statement_node",
	 bp::make_function(get_vcl_statement_node,
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
    
  // --------------------------------------------------

  // OpenCL interface

  /*

   + vcl::ocl::kernel
   + vcl::ocl::program
   + vcl::ocl::context

   PyOpenCL integration

   */

}

