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

typedef struct placesElement {
  vcl::vcl_size_t row;
  vcl::vcl_size_t col;
  placesElement* prev;
  placesElement* next;
  placesElement* last;
} placesElement;

template <class ScalarType>
class cpu_compressed_matrix_wrapper
{
  // TODO: This is just a quick first implementation. Later, I may well want 
  // TODO: a version that doesn't depend on boost.python types.
  typedef ublas::compressed_matrix<ScalarType, ublas::row_major> ublas_sparse_t;
  ublas_sparse_t cpu_compressed_matrix;

public:
  placesElement* places = NULL;

  cpu_compressed_matrix_wrapper()
  {
    cpu_compressed_matrix = ublas_sparse_t(0,0,0);
  }

  cpu_compressed_matrix_wrapper(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2)
  {
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2);
  }

  cpu_compressed_matrix_wrapper(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2, vcl::vcl_size_t _nnz)
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
    
    vcl::vcl_size_t n = array.shape(0);
    vcl::vcl_size_t m = array.shape(1);
    
    cpu_compressed_matrix = ublas_sparse_t(n, m);
    
    for (vcl::vcl_size_t i = 0; i < n; ++i) {
      for (vcl::vcl_size_t j = 0; j < m; ++j) {
	ScalarType val = bp::extract<ScalarType>(array[i][j]);
	if (val != 0)
          set_entry(i, j, val);
      }
    }
    
  }

  np::ndarray as_ndarray()
  {

    np::dtype dt = np::dtype::get_builtin<ScalarType>();
    bp::tuple shape = bp::make_tuple(size1(), size2());
    
    np::ndarray array = np::zeros(shape, dt);

    placesElement* place = places;
    while(place) {
      array[place->row][place->col] = get_entry(place->row, place->col);
      place = place->next;
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

    if (places)
      nnz(); // Make sure we have no bogus entries

    for (it1 i = cpu_compressed_matrix.begin1();
         i != cpu_compressed_matrix.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {

	if (cpu_compressed_matrix(j.index1(), j.index2())) {
          placesElement* place = (placesElement*)malloc(sizeof(placesElement));
          place->prev = NULL;
          place->next = NULL;
          place->last = place;
          place->row = j.index1();
          place->col = j.index2();

          if (places) {
            placesElement* last = places->last;
            place->prev = last;
            last->next = place;
            last->last = place;
          } else {
            places = place;
          }            
        }

      }
    }

  }

  vcl::vcl_size_t nnz()
  {
    vcl::vcl_size_t i = 0;  

    placesElement* place = places;
    while (place) {
      if (cpu_compressed_matrix(place->row, place->col) != 0) {
        i++;
        place = place->next;
      } else {
        // this place is 0, so we need to take it out of the list
        if (place->prev) {
          if (place->next) {
            place->prev->next = place->next;
            place->next->prev = place->prev;
          } else {
            place->prev->next = NULL;
            placesElement* prev = place->prev;
            while (prev) {
              prev->last = place->prev;
              prev = prev->prev;
            }
          }
        } else {
          if (place->next) {
            places = place->next;
            places->prev = NULL;
          } else {
            places = NULL;
          }
        }
        placesElement* next = place->next;
        free(place);
        place = next;
      }
    }

    return i;
  }

  bp::list places_to_python() 
  {
    bp::list pyplaces;
    placesElement* place = places;
    while (place) {
      // add place to pyplaces
      pyplaces.append(bp::make_tuple(place->row, place->col));
      place = place->next;
    }
    return pyplaces;
  }
      
  vcl::vcl_size_t size1() const
  {
    return cpu_compressed_matrix.size1();
  }

  vcl::vcl_size_t size2() const
  {
    return cpu_compressed_matrix.size2();
  }

  void resize(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2)
  {

    if ((_size1 == size1()) and (_size2 == size2()))
      return;

    // TODO NB: ublas compressed matrix does not support preserve on resize
    //          so this below is annoyingly hacky...

    cpu_compressed_matrix_wrapper temp(*this);
    cpu_compressed_matrix.resize(_size1, _size2, false); // preserve == false!

    placesElement* place = places;
    while (place) {
      if ((place->row < size1()) and (place->col < size2())) {
        cpu_compressed_matrix(place->row, place->col) = temp.get_entry(place->row, place->col);
      } else {
        // TODO: remove place from list
      }
      place = place->next;
    }

  }

  void set_entry(vcl::vcl_size_t n, vcl::vcl_size_t m, ScalarType val) 
  {
    if (n >= size1() or m >= size2())
      resize(n+1, m+1);

    // first we set the underlying entry
    cpu_compressed_matrix(n, m) = val;

    if (val != 0) {

      // then we need to make sure (n, m) is on the list of places
      placesElement* place = (placesElement*)malloc(sizeof(placesElement));
      place->prev = NULL;
      place->next = NULL;
      place->last = place;
      place->row = n;
      place->col = m;

      if (places) {
        // then we need to add (n, m) to the list
        place->prev = places->last;
        places->last->next = place;
        places->last->last = place;
        places->last = place;
      } else {
        // then we need to create a list and add (n, m) to it
        places = place;
      }

    } else if ((val == 0) && places) {
      printf("DEBUG (set_entry): %d, %d, %f\n", n, m, val);

      // then we need to make sure (n, m) is *not* on the list

      placesElement* next = places->next;

      while ((places->row == n) && (places->col == m)) {
        free(places);
        if (next) {
          places = next;
          places->prev = NULL;
          next = places->next;
        } else {
          places = NULL;
          break;
        }
      }

      placesElement* place = places;
      bool update_last = false;

      while (next) {
        if ((next->row == n) && (next->col == m)) {
          next->next->prev = place;
          place->next = next->next;
          if (!(place->next)) {
            update_last = true;
          }
          free(next);
        } else {
          place = next;
        }
        next = place->next;
      }
      
      if (update_last) {
        placesElement* prev = place->prev;
        while (prev) {
          prev->last = place;
          prev = prev->prev;
        }
      }        

    }
          
  }

  // Need this because bp cannot deal with operator()
  ScalarType get_entry(vcl::vcl_size_t n, vcl::vcl_size_t m)
  {
    return cpu_compressed_matrix(n, m);
  }

};

#endif
