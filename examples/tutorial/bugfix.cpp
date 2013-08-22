#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>

#include "viennacl/matrix.hpp"
#include "viennacl/scheduler/execute.hpp"
#include "viennacl/scheduler/io.hpp"

int main() {
  
  boost::numeric::ublas::matrix<double> cpu_m(131, 131);

  for (unsigned int i = 0; i < cpu_m.size1(); ++i) {
    for (unsigned int j = 0; j < cpu_m.size2(); ++j)
      cpu_m(i, j) = (double) (3.142 * i + j);
  }
  
  viennacl::matrix<double> n(131, 131);
  viennacl::matrix<double> m(131, 131);
  viennacl::copy(cpu_m, m);
  
  typedef viennacl::scheduler::statement::container_type con_t;
  
  // Want an expression for ElementFabs(Sub(Matrix, Mul(Matrix, Matrix)))
  //  so, four nodes (including an implicit Assign)
  
  con_t expr(8);
  
  expr[0].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[0].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[0].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[0].lhs.matrix_row_double = &n;
  expr[0].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[0].op.type = viennacl::scheduler::OPERATION_BINARY_ASSIGN_TYPE;
  expr[0].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[0].rhs.node_index = 1;
  
  expr[1].lhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[1].lhs.node_index = 2;
  expr[1].op.type_family = viennacl::scheduler::OPERATION_UNARY_TYPE_FAMILY;
  expr[1].op.type = viennacl::scheduler::OPERATION_UNARY_FABS_TYPE;
  //expr[1].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  //expr[1].rhs.node_index = 2;
  
  expr[2].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[2].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[2].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[2].lhs.matrix_row_double = &m;
  expr[2].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[2].op.type = viennacl::scheduler::OPERATION_BINARY_SUB_TYPE;
  expr[2].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr[2].rhs.node_index = 3;
  
  expr[3].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[3].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[3].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[3].lhs.matrix_row_double = &m;
  expr[3].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[3].op.type = viennacl::scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE;
  expr[3].rhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[3].rhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr[3].rhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[3].rhs.matrix_row_double = &m;
  
  viennacl::scheduler::statement test(expr);
  
  //std::cout << test << std::endl;

  viennacl::scheduler::execute(test);
  
  std::cout << n << std::endl;

  return EXIT_SUCCESS;

}

