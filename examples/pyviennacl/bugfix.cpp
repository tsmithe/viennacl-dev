#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <stdint.h>

#define VIENNACL_WITH_OPENCL
#include "viennacl/matrix.hpp"
#include "viennacl/scheduler/execute.hpp"
#include "viennacl/scheduler/io.hpp"

namespace ublas = boost::numeric::ublas;

int main() {
  
  typedef viennacl::scheduler::statement::container_type con_t;

  const std::size_t x = 10, y = 10;

  ublas::matrix<double> cpu_m(x, y);

  for (std::size_t i = 0; i < cpu_m.size1(); ++i) {
    for (std::size_t j = 0; j < cpu_m.size2(); ++j)
      cpu_m(i, j) = (double) (i*cpu_m.size2() + j);
  }

  viennacl::matrix<double> m(x, y);
  viennacl::matrix<double> n(x, y);

  viennacl::copy(cpu_m, m);

  con_t expr(5);

  // n = [1]
  expr[0].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[0].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[0].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[0].lhs.matrix_row_double = &n;
  expr[0].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[0].op.type = viennacl::scheduler::OPERATION_BINARY_ASSIGN_TYPE;
  expr[0].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[0].rhs.node_index = 1;

  // [1] := fabs([2])
  expr[1].lhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[1].lhs.node_index = 2;
  expr[1].op.type_family = viennacl::scheduler::OPERATION_UNARY_TYPE_FAMILY;
  expr[1].op.type = viennacl::scheduler::OPERATION_UNARY_FABS_TYPE;

  // [2] := m - [3]
  expr[2].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[2].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[2].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[2].lhs.matrix_row_double = &m;
  expr[2].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[2].op.type = viennacl::scheduler::OPERATION_BINARY_SUB_TYPE;
  expr[2].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[2].rhs.node_index = 3;

  // [3] := m + [4]
  expr[3].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[3].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[3].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[3].lhs.matrix_row_double = &m;
  expr[3].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[3].op.type = viennacl::scheduler::OPERATION_BINARY_ADD_TYPE;
  expr[3].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[3].rhs.node_index = 4;

  // [4] := m * 2.718
  expr[4].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[4].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[4].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[4].lhs.matrix_row_double = &m;
  expr[4].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[4].op.type = viennacl::scheduler::OPERATION_BINARY_MULT_TYPE;
  expr[4].rhs.type_family = viennacl::scheduler::SCALAR_TYPE_FAMILY;
  expr[4].rhs.subtype = viennacl::scheduler::HOST_SCALAR_TYPE;
  expr[4].rhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[4].rhs.host_double = 2.718;
  
  // n = fabs(m - (m + (m * 2.718)))
  viennacl::scheduler::statement test(expr);
  
  std::cout << test << std::endl;

  viennacl::scheduler::execute(test);

  return EXIT_SUCCESS;

}

