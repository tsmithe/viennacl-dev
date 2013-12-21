/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#ifndef NDEBUG
  #define NDEBUG
#endif

//
// *** System
//
#include <iostream>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "examples/tutorial/Random.hpp"
#include "viennacl/generator/generate.hpp"
#include "list"
//
// -------------------------------------------------------------
//
using namespace boost::numeric;
//
// -------------------------------------------------------------
//
static const unsigned int min_large_block_size = 32;
static const unsigned int max_large_block_size = 128;
static const unsigned int n_large_blocks = static_cast<unsigned int>(std::log(static_cast<double>(max_large_block_size/min_large_block_size))/std::log(2.0)+1.0);

static const unsigned int min_alignment = 1;
static const unsigned int max_alignment = 8;

static const unsigned int max_small_block_size = max_alignment;

//
// -------------------------------------------------------------

template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2)
{
   viennacl::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}

template <typename ScalarType, typename VCLMatrixType>
ScalarType diff(ublas::matrix<ScalarType> & mat1, VCLMatrixType & mat2)
{
   ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(mat2, mat2_cpu);
   ScalarType ret = 0;
   ScalarType act = 0;

    for (unsigned int i = 0; i < mat2_cpu.size1(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu.size2(); ++j)
      {
         act = std::fabs(mat2_cpu(i,j) - mat1(i,j)) / std::max( std::fabs(mat2_cpu(i, j)), std::fabs(mat1(i,j)) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
}






//
// Part 1: Matrix-matrix multiplications
//


template< typename NumericT, typename Epsilon,
          typename ReferenceMatrixTypeA, typename ReferenceMatrixTypeB, typename ReferenceMatrixTypeC,
          typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
int test_prod(Epsilon const& epsilon,

              ReferenceMatrixTypeA const & A, ReferenceMatrixTypeA const & A_trans,
              ReferenceMatrixTypeB const & B, ReferenceMatrixTypeB const & B_trans,
              ReferenceMatrixTypeC & C,

              MatrixTypeA const & vcl_A, MatrixTypeA const & vcl_A_trans,
              MatrixTypeB const & vcl_B, MatrixTypeB const & vcl_B_trans,
              MatrixTypeC & vcl_C
             )
{
   int retval = EXIT_SUCCESS;
   NumericT act_diff = 0;
   NumericT alpha = NumericT(3.14);
   NumericT beta  = NumericT(4.51);

std::cout << "Testing C = alpha*prod(A,B) + beta*C ..." << std::endl;
{
    C     = alpha*viennacl::linalg::prod(A, B) + beta*C;

    viennacl::scheduler::statement statement(vcl_C, viennacl::op_assign(), alpha*viennacl::linalg::prod(vcl_A,vcl_B)+beta*vcl_C);
    viennacl::generator::generate_enqueue_statement(statement, statement.array()[0]);
    viennacl::backend::finish();
    act_diff = std::fabs(diff(C, vcl_C));
    if( act_diff > epsilon )
    {
      std::cout << "# Error at operation: matrix-matrix product" << std::endl;
      std::cout << "  diff: " << act_diff << std::endl;
      retval = EXIT_FAILURE;
    }
    else
      std::cout << "Test C = A * B passed!" << std::endl;
}


   std::cout << "Testing C = alpha*trans(A) * B + beta*C ..." << std::endl;
   {
       C     = alpha*boost::numeric::ublas::prod(trans(A_trans), B) + beta*C;
       viennacl::scheduler::statement statement(vcl_C, viennacl::op_assign(), alpha*viennacl::linalg::prod(trans(vcl_A_trans),vcl_B) + beta*vcl_C);
       viennacl::generator::generate_enqueue_statement(statement, statement.array()[0]);
       viennacl::backend::finish();
       act_diff = std::fabs(diff(C, vcl_C));
       if( act_diff > epsilon )
       {
         std::cout << "# Error at operation: matrix-matrix product" << std::endl;
         std::cout << "  diff: " << act_diff << std::endl;
         retval = EXIT_FAILURE;
       }
       else std::cout << "Test C = trans(A) * B passed!" << std::endl;
   }

std::cout << "Testing C = alpha*A * trans(B) + beta*C ..." << std::endl;
{
    C     = boost::numeric::ublas::prod(A,trans(B_trans)) + beta*C;
    viennacl::scheduler::statement statement(vcl_C, viennacl::op_assign(), viennacl::linalg::prod(vcl_A,trans(vcl_B_trans)) + beta*vcl_C);
    viennacl::generator::generate_enqueue_statement(statement, statement.array()[0]);
    viennacl::backend::finish();
    act_diff = std::fabs(diff(C, vcl_C));
    if( act_diff > epsilon )
    {
      std::cout << "# Error at operation: matrix-matrix product" << std::endl;
      std::cout << "  diff: " << act_diff << std::endl;
      retval = EXIT_FAILURE;
    }
    else std::cout << "Test C = A * trans(B) passed!" << std::endl;
}

std::cout << "Testing C = alpha*trans(A) * trans(B) + beta*C ..." << std::endl;
{
    C     = boost::numeric::ublas::prod(trans(A_trans), trans(B_trans)) + beta*C;
    viennacl::scheduler::statement statement(vcl_C, viennacl::op_assign(), viennacl::linalg::prod(trans(vcl_A_trans),trans(vcl_B_trans)) + beta*vcl_C);
    viennacl::generator::generate_enqueue_statement(statement, statement.array()[0]);
    viennacl::backend::finish();
    act_diff = std::fabs(diff(C, vcl_C));
    if( act_diff > epsilon )
    {
      std::cout << "# Error at operation: matrix-matrix product" << std::endl;
      std::cout << "  diff: " << act_diff << std::endl;
      retval = EXIT_FAILURE;
    }
    else std::cout << "Test C = trans(A) * trans(B) passed!" << std::endl;
}


   return retval;
}

template< typename NumericT, typename F_A, typename F_B, typename F_C, typename Epsilon>
int test_prod(Epsilon const& epsilon)
{
  int ret;

  long matrix_size1 = 2*max_large_block_size;
  long matrix_size2 = 3*max_large_block_size;
  long matrix_size3 = 4*max_large_block_size;

  // --------------------------------------------------------------------------

  // ublas reference:
  ublas::matrix<NumericT> A(matrix_size1, matrix_size2);
  ublas::matrix<NumericT> B(matrix_size2, matrix_size3);
  ublas::matrix<NumericT> C(matrix_size1, matrix_size3);

  //fill A and B:
  for (unsigned int i = 0; i < A.size1(); ++i)
    for (unsigned int j = 0; j < A.size2(); ++j)
        A(i,j) = static_cast<NumericT>(0.1) * random<NumericT>();
  for (unsigned int i = 0; i < B.size1(); ++i)
    for (unsigned int j = 0; j < B.size2(); ++j)
        B(i,j) = static_cast<NumericT>(0.1) * random<NumericT>();
  for (unsigned int i = 0; i < C.size1(); ++i)
    for (unsigned int j = 0; j < C.size2(); ++j)
        C(i,j) = static_cast<NumericT>(0.1) * random<NumericT>();


  ublas::matrix<NumericT>     A_trans = trans(A);
  ublas::matrix<NumericT>     B_trans = trans(B);

  //
  // ViennaCL objects
  //



  // A
  viennacl::matrix<NumericT, F_A>    vcl_A(matrix_size1, matrix_size2);
  viennacl::copy(A, vcl_A);

  // A^T
  viennacl::matrix<NumericT, F_A>    vcl_A_trans(matrix_size2, matrix_size1);
  viennacl::copy(A_trans, vcl_A_trans);

  // B
  viennacl::matrix<NumericT, F_B>    vcl_B(matrix_size2, matrix_size3);
  viennacl::copy(B, vcl_B);

  // B^T
  viennacl::matrix<NumericT, F_B>    vcl_B_trans(matrix_size3, matrix_size2);
  viennacl::copy(B_trans, vcl_B_trans);

  // C
  viennacl::matrix<NumericT, F_C>    vcl_C(matrix_size1, matrix_size3);
  viennacl::copy(C, vcl_C);

  std::cout << "--- Part 1: Testing matrix-matrix products ---" << std::endl;

  //////
  //////  A: matrix
  //////

  //
  //
  std::cout << "Now using A=matrix, B=matrix, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            vcl_A, vcl_A_trans,
                            vcl_B, vcl_B_trans,
                            vcl_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  return EXIT_SUCCESS;
}

template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=row, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::row_major, viennacl::row_major, viennacl::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=row, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::column_major, viennacl::row_major, viennacl::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=col, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::row_major, viennacl::column_major, viennacl::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=col, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::column_major, viennacl::column_major, viennacl::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;


  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=row, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::row_major, viennacl::row_major, viennacl::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=row, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::column_major, viennacl::row_major, viennacl::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=col, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::row_major, viennacl::column_major, viennacl::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=col, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, viennacl::column_major, viennacl::column_major, viennacl::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;



  return ret;
}

int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv,argv+argc);
    int retval = EXIT_SUCCESS;

    typedef std::vector<viennacl::ocl::device> devices_type;

    //platforms_type platforms = viennacl::ocl::get_platforms();
    //size_t num_platforms = platforms.size();

    devices_type dev = viennacl::ocl::current_context().devices();
    for(devices_type::iterator it = dev.begin() ; it != dev.end() ; ++it){
            std::cout << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            std::cout << "## Test :: Generated BLAS 3 routines" << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            std::cout << std::endl;

            int retval = EXIT_SUCCESS;

            //srand(time(NULL));

            std::cout << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            std::cout << std::endl;
            {
               typedef float NumericT;
               NumericT epsilon = NumericT(1.0E-3);
               std::cout << "# Testing setup:" << std::endl;

               std::cout << viennacl::ocl::current_device().info() << std::endl;

               std::cout << "  eps:     " << epsilon << std::endl;
               std::cout << "  numeric: float" << std::endl;
               retval = test<NumericT>(epsilon);
               if( retval == EXIT_SUCCESS )
                 std::cout << "# Test passed" << std::endl;
               else
                 return retval;
            }
            std::cout << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            std::cout << std::endl;
         #ifdef VIENNACL_WITH_OPENCL
            if( viennacl::ocl::current_device().double_support() )
         #endif
            {
               {
                 typedef double NumericT;
                 NumericT epsilon = 1.0E-11;
                 std::cout << "# Testing setup:" << std::endl;
                 std::cout << "  eps:     " << epsilon << std::endl;
                 std::cout << "  numeric: double" << std::endl;
                 retval = test<NumericT>(epsilon);
                 if( retval == EXIT_SUCCESS )
                   std::cout << "# Test passed" << std::endl;
                 else
                   return retval;
               }
               std::cout << std::endl;
               std::cout << "----------------------------------------------" << std::endl;
               std::cout << std::endl;
            }

            std::cout << std::endl;
            std::cout << "------- Test completed --------" << std::endl;
            std::cout << std::endl;
    }




   return retval;
}

