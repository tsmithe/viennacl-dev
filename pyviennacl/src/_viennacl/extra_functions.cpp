#include "viennacl.h"

#define EXPORT_FUNCTIONS(TYPE, F)                                       \
  bp::def("outer", pyvcl_do_2ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_outer_prod, 0>);                                           \
  bp::def("element_pow", pyvcl_do_2ary_op<vcl::vector<TYPE>,            \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_element_pow, 0>);                                          \
  bp::def("element_pow", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,         \
          vcl::matrix_base<TYPE, F>&, vcl::matrix_base<TYPE, F>&,       \
          op_element_pow, 0>);                                          \
  bp::def("plane_rotation", pyvcl_do_4ary_op<bp::object,                \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          TYPE, TYPE,                                                   \
          op_plane_rotation, 0>);                                  

PYVCL_MODULE(extra_functions)

  // vector: outer, element_pow, plane_rotation
  // matrix: element_pow

  // TODO missing: char, short, uchar, ushort
  EXPORT_FUNCTIONS(int, vcl::row_major)
  EXPORT_FUNCTIONS(int, vcl::column_major)
  EXPORT_FUNCTIONS(long, vcl::row_major)
  EXPORT_FUNCTIONS(long, vcl::column_major)
  EXPORT_FUNCTIONS(uint, vcl::row_major)
  EXPORT_FUNCTIONS(uint, vcl::column_major)
  EXPORT_FUNCTIONS(ulong, vcl::row_major)
  EXPORT_FUNCTIONS(ulong, vcl::column_major)
  EXPORT_FUNCTIONS(double, vcl::row_major)
  EXPORT_FUNCTIONS(double, vcl::column_major)
  EXPORT_FUNCTIONS(float, vcl::row_major)
  EXPORT_FUNCTIONS(float, vcl::column_major)

}
