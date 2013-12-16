#include "_viennacl.h"

void export_dense_matrix_ulong() {
  EXPORT_DENSE_MATRIX_CLASS(ulong, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(ulong, col, vcl::column_major, ublas::column_major)
}

