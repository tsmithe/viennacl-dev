#include "_viennacl.h"

void export_dense_matrix_int() {
  EXPORT_DENSE_MATRIX_CLASS(int, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(int, col, vcl::column_major, ublas::column_major)
}

