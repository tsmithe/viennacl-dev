#ifndef _PYVIENNACL_VECTOR_H
#define _PYVIENNACL_VECTOR_H

#include "viennacl.h"

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>

#include <viennacl/linalg/sparse_matrix_operations.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>

namespace ublas = boost::numeric::ublas;

// TODO: EXPOSE ALL NUMERIC TYPES

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
      array[x][y] = get_entry(x, y);
    }

    return array;

  }

  template<class SparseT>
  vcl::tools::shared_ptr<SparseT>
  as_vcl_sparse_matrix()
  {
    SparseT* vcl_sparse_matrix = new SparseT();
    vcl::copy(cpu_compressed_matrix, *vcl_sparse_matrix);
    return vcl::tools::shared_ptr<SparseT>(vcl_sparse_matrix);
  }

  template<class SparseT>
  vcl::tools::shared_ptr<SparseT>
  as_vcl_sparse_matrix_with_size()
  {
    SparseT* vcl_sparse_matrix = new SparseT(size1(), size2(), nnz());
    vcl::copy(cpu_compressed_matrix, *vcl_sparse_matrix);
    return vcl::tools::shared_ptr<SparseT>(vcl_sparse_matrix);
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
      cpu_compressed_matrix(x, y) = temp.get_entry(x, y);
    }

  }

  void set_entry(uint32_t n, uint32_t m, ScalarType val) 
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
  ScalarType get_entry(uint32_t n, uint32_t m)
  {
    return cpu_compressed_matrix(n, m);
  }

};

#endif
