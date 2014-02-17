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
  placesElement* places = NULL;

public:
  void print_places() {
    placesElement* place = places;
    while (place) {
      printf("%8p = {prev:  %8p\
    next:  %8p\
    last:  %8p\
    row:   %8lu\
    col:   %8lu}\n",
             place, place->prev, place->next, place->last,
             place->row, place->col);
      place = place->next;
    }
  }

  void add_place(vcl::vcl_size_t row, vcl::vcl_size_t col) {

    //printf("DEBUG: add_place(%d, %d)\n", row, col);

    placesElement* place = (placesElement*)malloc(sizeof(placesElement));
    place->prev = NULL;
    place->next = NULL;
    place->last = place;
    place->row = row;
    place->col = col;
    
    if (places) {
      placesElement* last = places->last;
      place->prev = last;
      last->next = place;
      while (last) {
        last->last = place;
        last = last->prev;
      }
    } else {
      places = place;
    }            

    //print_places();
    //printf("/// add_place(**)\n");

  }

  void remove_place(vcl::vcl_size_t row, vcl::vcl_size_t col, bool only_dupes) {
    if (!places)
      return;

    //printf("DEBUG: remove_place(%d, %d, %d)\n", row, col, only_dupes);

    placesElement* next = places->next;
    bool got_one = false;

    while ((places->row == row) and (places->col == col)) {
      if (got_one or (!only_dupes)) {
        //if (only_dupes)
        //  printf("remove_place: FOUND DUPE of (%d, %d) at %p (places)\n", row, col, places);
        free(places);
        places = next;
        if (next)
          places->prev = NULL;
        else
          break;
      } else {
        //printf("remove_place: place (%d, %d) at %p (places)\n", row, col, places);
        got_one = true;
        if (only_dupes)
          break;
      }
      next = places->next;
    }
    
    placesElement* place = places;
    bool update_last = false;
    
    while (next) {
      if ((next->row == row) && (next->col == col)) {
        if (got_one or (!only_dupes)) {
          //if (only_dupes)
          //  printf("remove_place: FOUND DUPE of (%d, %d) at %p (next)\n", row, col, next);
          if (next->next)
            next->next->prev = place;
          if (place->next)
            place->next = next->next;
          else
            update_last = true;
          free(next);
        } else {
          //printf("remove_place: place (%d, %d) at %p (next)\n", row, col, next);
          got_one = true;
          place = next;
        }
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

    //print_places();
    //printf("/// remove_place(***)\n");

  }

  void remove_place(vcl::vcl_size_t row, vcl::vcl_size_t col) {
    remove_place(row, col, false);
  }

  void remove_place(placesElement* place) {

    //printf("DEBUG: remove_place(%p)\n", place);

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
    free(place);

    //print_places();
    //printf("/// remove_place(*)\n");

  }

  vcl::vcl_size_t check_places() {
    vcl::vcl_size_t i = 0;  

    //printf("DEBUG (check_places)...\n");

    placesElement* place = places;
    while (place) {
      ScalarType val = cpu_compressed_matrix(place->row, place->col);      
      //printf("check_places: %p, %d, %d, %f\n", place, place->row, place->col, val);
      placesElement* next = place->next;
      if (val != 0) {
        i++;
        remove_place(place->row, place->col, true); // remove duplicate places
      } else {
        remove_place(place); // remove all entries
      }
      place = next;
    }

    //printf("/// check_places(%d)\n", i);

    return i;
  }

  vcl::vcl_size_t update_places()
  {

    printf("DEBUG (update_places)...\n");

    typedef typename ublas_sparse_t::iterator1 it1;
    typedef typename ublas_sparse_t::iterator2 it2;

    vcl::vcl_size_t i = 0;

    for (it1 i = cpu_compressed_matrix.begin1();
         i != cpu_compressed_matrix.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {

	if (cpu_compressed_matrix(j.index1(), j.index2()) != 0) {
          //printf("update_places: found place at (%d, %d)\n", j.index1(), j.index2());
          i++;
          //add_place(j.index1(), j.index2());
        }

      }
    }

    printf("/// update_places\n");

    return i; //check_places(); // Make sure we have no bogus entries

  }

  bp::list places_to_python() 
  {
    //update_places();

    bp::list pyplaces;

    /*
    placesElement* place = places;
    while (place) {
      // add place to pyplaces
      pyplaces.append(bp::make_tuple(place->row, place->col));
      place = place->next;
    }
    */

    typedef typename ublas_sparse_t::iterator1 it1;
    typedef typename ublas_sparse_t::iterator2 it2;

    for (it1 i = cpu_compressed_matrix.begin1();
         i != cpu_compressed_matrix.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {

	if (cpu_compressed_matrix(j.index1(), j.index2()) != 0) {
          pyplaces.append(bp::make_tuple(j.index1(), j.index2()));
        }

      }
    }

    return pyplaces;
  }

  cpu_compressed_matrix_wrapper()
  {
    //printf("DEBUG: constructor()\n");
    cpu_compressed_matrix = ublas_sparse_t(0,0,0);
    //printf("/// constructor()\n");
  }

  cpu_compressed_matrix_wrapper(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2)
  {
    //printf("DEBUG: constructor(%d, %d)\n", _size1, _size2);
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2);
    //printf("/// constructor(**)\n");
  }

  cpu_compressed_matrix_wrapper(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2, vcl::vcl_size_t _nnz)
  {
    //printf("DEBUG: constructor(%d, %d, %d)\n", _size1, _size2, _nnz);
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2, _nnz);
    //printf("/// constructor(***)\n");
  }

  cpu_compressed_matrix_wrapper(const cpu_compressed_matrix_wrapper& w)
    : cpu_compressed_matrix(w.cpu_compressed_matrix)
  {
    //printf("DEBUG: constructor(const& cpu %p) with size (%d, %d)\n", &w, size1(), size2());
    //update_places();
    //printf("/// constructor(*)\n");
  }

  template<class SparseT>
  cpu_compressed_matrix_wrapper(const SparseT& vcl_sparse_matrix)
  {
    //printf("DEBUG: constructor(const& vcl %p)\n", &vcl_sparse_matrix);
    cpu_compressed_matrix = ublas_sparse_t(vcl_sparse_matrix.size1(),
                                           vcl_sparse_matrix.size2());
    vcl::copy(vcl_sparse_matrix, cpu_compressed_matrix);
    
    //update_places();
    //printf("/// constructor(*)\n");
  }

  cpu_compressed_matrix_wrapper(const np::ndarray& array)
  {
    //printf("DEBUG: constructor(const& ndarray %p)\n", &array);

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

    //printf("/// constructor(*)\n");
  }

  np::ndarray as_ndarray()
  {

    update_places();

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

  vcl::vcl_size_t nnz()
  {
    return update_places();
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

    //printf("DEBUG: resize(%d, %d) with size (%d, %d)\n", 
    //       _size1, _size2, size1(), size2());

    if ((_size1 == size1()) and (_size2 == size2()))
      return;

    // TODO NB: ublas compressed matrix does not support preserve on resize
    //          so this below is annoyingly hacky...

    ublas_sparse_t temp(cpu_compressed_matrix); // Incurs a copy of all the data!!
    cpu_compressed_matrix.resize(_size1, _size2, false); // preserve == false!

    /*
    placesElement* place = places;
    while (place) {
      if ((place->row < size1()) and (place->col < size2())) {
        cpu_compressed_matrix(place->row, place->col) = temp(place->row, place->col);
        place = place->next;
      } else {
        placesElement* next = place->next;
        remove_place(place);
        place = next;
      }
    }
    */

    //printf("/// resize(**)\n");
  }
  
  void set_entry(vcl::vcl_size_t n, vcl::vcl_size_t m, ScalarType val) 
  {
    if (n >= size1()) {
      if (m >= size2())
        resize(n+1, m+1);
      else
        resize(n+1, size2());
    } else {
      if (m >= size2())
        resize(size1(), m+1);
    }

    //printf("DEBUG: set_entry(%d, %d, %f)\n", n, m, val);

    // first we set the underlying entry
    cpu_compressed_matrix(n, m) = val;

    //if (val != 0)
    //  add_place(n, m);
    //else
    //  remove_place(n, m);
    
    //printf("/// set_entry(***)\n");

  }

  // Need this because bp cannot deal with operator()
  ScalarType get_entry(vcl::vcl_size_t n, vcl::vcl_size_t m)
  {
    return cpu_compressed_matrix(n, m);
  }

};

#endif
