#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <stdint.h>

#define VIENNACL_WITH_OPENCL
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/scheduler/execute.hpp"
#include "viennacl/scheduler/io.hpp"

int main() {
  
  std::size_t x = 15, y = 15;

  boost::numeric::ublas::matrix<double> cpu_m(x, y);

  for (std::size_t i = 0; i < cpu_m.size1(); ++i) {
    for (std::size_t j = 0; j < cpu_m.size2(); ++j)
      cpu_m(i, j) = (double) (3.142 * i + j);
  }

  typedef viennacl::scheduler::statement::container_type con_t;

  /*
  
  viennacl::matrix<double> k(x, y);
  viennacl::matrix<double> l(x, y);
  viennacl::matrix<double> m(x, y);
  viennacl::matrix<double> n(x, y);
  viennacl::matrix<double> o(x, y);
  viennacl::copy(cpu_m, m);
  cpu_m *= 2.178;
  viennacl::copy(cpu_m, l);
  k = l - viennacl::linalg::prod(m, m);

  std::cout << "k = " << k << std::endl;
  
  // Want an expression for ElementFabs(Sub(Matrix, Mul(Matrix, Matrix)))
  //  so, four nodes (including an implicit Assign)
  
  con_t expr1(8);
  
  expr1[0].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr1[0].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr1[0].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr1[0].lhs.matrix_row_double = &n;
  expr1[0].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr1[0].op.type = viennacl::scheduler::OPERATION_BINARY_ASSIGN_TYPE;
  expr1[0].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr1[0].rhs.node_index = 1;
  
  expr1[1].lhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr1[1].lhs.node_index = 2;
  expr1[1].op.type_family = viennacl::scheduler::OPERATION_UNARY_TYPE_FAMILY;
  expr1[1].op.type = viennacl::scheduler::OPERATION_UNARY_FABS_TYPE;
  expr1[1].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr1[1].rhs.node_index = 2;
  
  expr1[2].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr1[2].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr1[2].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr1[2].lhs.matrix_row_double = &l;
  expr1[2].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr1[2].op.type = viennacl::scheduler::OPERATION_BINARY_SUB_TYPE;
  expr1[2].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr1[2].rhs.node_index = 3;
  
  expr1[3].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr1[3].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr1[3].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr1[3].lhs.matrix_row_double = &m;
  expr1[3].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr1[3].op.type = viennacl::scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE;
  expr1[3].rhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr1[3].rhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr1[3].rhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr1[3].rhs.matrix_row_double = &m;
  
  viennacl::scheduler::statement test1(expr1);
  
  std::cout << test1 << std::endl;

  viennacl::scheduler::execute(test1);
  
  std::cout << "n = " << n << std::endl;

  con_t expr2(2);
  
  expr2[0].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr2[0].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr2[0].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr2[0].lhs.matrix_row_double = &o;
  expr2[0].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr2[0].op.type = viennacl::scheduler::OPERATION_BINARY_ASSIGN_TYPE;
  expr2[0].rhs.type_family = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY;
  expr2[0].rhs.node_index = 1;
  
  expr2[1].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr2[1].lhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr2[1].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr2[1].lhs.matrix_row_double = &k;
  expr2[1].op.type_family = viennacl::scheduler::OPERATION_UNARY_TYPE_FAMILY;
  expr2[1].op.type = viennacl::scheduler::OPERATION_UNARY_FABS_TYPE;
  expr2[1].rhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr2[1].rhs.subtype = viennacl::scheduler::DENSE_ROW_MATRIX_TYPE;
  expr2[1].rhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr2[1].rhs.matrix_row_double = &k;

  viennacl::scheduler::statement test2(expr2);
  
  std::cout << test2 << std::endl;

  viennacl::scheduler::execute(test2);
  
  std::cout << "o = " << o << std::endl;

  viennacl::matrix<double> p(x, y);
  p = n - o;

  std::cout << "p = " << p << std::endl;

  */

  viennacl::matrix<double, viennacl::column_major> m(x, y);

  viennacl::copy(cpu_m, m);

  viennacl::range range1(1, 5);
  viennacl::range range2(2, 6);

  viennacl::matrix_range<viennacl::matrix<double, viennacl::column_major> >
    m_range(m, range1, range2);

  viennacl::matrix<double, viennacl::column_major> n(m_range);

  con_t expr(1);

  std::cout << "n = " << n << std::endl;
  
  expr[0].lhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[0].lhs.subtype = viennacl::scheduler::DENSE_COL_MATRIX_TYPE;
  expr[0].lhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[0].lhs.matrix_col_double = &n;
  expr[0].op.type_family = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY;
  expr[0].op.type = viennacl::scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE;
  expr[0].rhs.type_family = viennacl::scheduler::MATRIX_TYPE_FAMILY;
  expr[0].rhs.subtype = viennacl::scheduler::DENSE_COL_MATRIX_TYPE;
  expr[0].rhs.numeric_type = viennacl::scheduler::DOUBLE_TYPE;
  expr[0].rhs.matrix_col_double = &n;
  
  viennacl::scheduler::statement test(expr);
  
  std::cout << test << std::endl;

  viennacl::scheduler::execute(test);
  
  std::cout << "n = " << n << std::endl;

  std::cout << "m = " << m << std::endl;

  return EXIT_SUCCESS;

}

