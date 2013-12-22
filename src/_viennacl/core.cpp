#include <stdint.h>
#include <iostream>
#include <typeinfo>

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS
#define VIENNACL_WITH_OPENCL

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/linalg/gmres.hpp>

#include "viennacl.h"

/*******************************
  Python module initialisation
 *******************************/

PYVCL_MODULE(core)
  bp::def("backend_finish", vcl::backend::finish);

  // TODO: EXPOSE ALL NUMERIC TYPES

  bp::class_<vcl::scalar<float> >("scalar_float") // TODO
    .def(bp::init<float>())
    .def(bp::init<int>())
    .def("to_host", &vcl_scalar_to_host<float>)
    ;

  bp::class_<vcl::scalar<double> >("scalar_double")
    .def(bp::init<double>())
    .def(bp::init<int>())
    .def("to_host", &vcl_scalar_to_host<double>)
    ;

  bp::class_<vcl::range>("range",
                         bp::init<std::size_t, std::size_t>());
  bp::class_<vcl::slice>("slice",
                         bp::init<std::size_t, std::size_t, std::size_t>());

  //EXPORT_VECTOR_CLASS(char)
  //EXPORT_VECTOR_CLASS(short)

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

  /* TODO:::::::::::::::
  EXPORT_DENSE_MATRIX_CLASS(char, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(char, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(short, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(short, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(uchar, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(uchar, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(ushort, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(ushort, col, vcl::column_major, ublas::column_major)
  */
  
}

