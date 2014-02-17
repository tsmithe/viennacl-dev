#ifndef _PYVIENNACL_ITERATIVE_SOLVERS_H
#define _PYVIENNACL_ITERATIVE_SOLVERS_H

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/matrix.hpp>

#include "viennacl.h"
#include "solve_op_func.hpp"

#define EXPORT_ITERATIVE_SOLVERS_F(TYPE, F)                             \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::cg_tag&,                                         \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::bicgstab_tag&,                                   \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::gmres_tag&,                                      \
          op_solve, 0>);

#define EXPORT_ITERATIVE_SOLVERS(TYPE)                  \
  EXPORT_ITERATIVE_SOLVERS_F(TYPE, vcl::row_major);     \
  EXPORT_ITERATIVE_SOLVERS_F(TYPE, vcl::column_major);

#endif
