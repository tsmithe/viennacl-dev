#ifndef VIENNACL_LINALG_OPENCL_DIRECT_SOLVE_HPP
#define VIENNACL_LINALG_OPENCL_DIRECT_SOLVE_HPP

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

/** @file viennacl/linalg/opencl/direct_solve.hpp
    @brief Implementations of dense direct solvers are found here.
*/

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/linalg/opencl/kernels/matrix_solve.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      namespace detail
      {
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::upper_tag)      { return 0; }
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::unit_upper_tag) { return (1 << 0); }
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::lower_tag)      { return (1 << 2); }
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::unit_lower_tag) { return (1 << 2) | (1 << 0); }

        template <typename M1, typename M2, typename KernelType>
        void inplace_solve_impl(M1 const & A, M2 & B, KernelType & k)
        {
          viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(A),
                                   cl_uint(viennacl::traits::start1(A)),         cl_uint(viennacl::traits::start2(A)),
                                   cl_uint(viennacl::traits::stride1(A)),        cl_uint(viennacl::traits::stride2(A)),
                                   cl_uint(viennacl::traits::size1(A)),          cl_uint(viennacl::traits::size2(A)),
                                   cl_uint(viennacl::traits::internal_size1(A)), cl_uint(viennacl::traits::internal_size2(A)),
                                   viennacl::traits::opencl_handle(B),
                                   cl_uint(viennacl::traits::start1(B)),         cl_uint(viennacl::traits::start2(B)),
                                   cl_uint(viennacl::traits::stride1(B)),        cl_uint(viennacl::traits::stride2(B)),
                                   cl_uint(viennacl::traits::size1(B)),          cl_uint(viennacl::traits::size2(B)),
                                   cl_uint(viennacl::traits::internal_size1(B)), cl_uint(viennacl::traits::internal_size2(B))
                                  )
                                );
        }
      }


      //
      // Note: By convention, all size checks are performed in the calling frontend. No need to double-check here.
      //

      ////////////////// upper triangular solver (upper_tag) //////////////////////////////////////
      /** @brief Direct inplace solver for dense triangular systems. Matlab notation: A \ B
      *
      * @param A    The system matrix
      * @param B    The matrix of row vectors, where the solution is directly written to
      */
      template <typename NumericT, typename SOLVERTAG>
      void inplace_solve(const matrix_base<NumericT> & A, bool trans_A,
                         matrix_base<NumericT> & B, bool trans_B,
                         SOLVERTAG)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

        std::string program_name;
        if (A.row_major() && B.row_major())
        {
          typedef viennacl::linalg::opencl::kernels::matrix_solve<NumericT, row_major, row_major>    KernelClass;
          KernelClass::init(ctx);
          program_name = KernelClass::program_name();
        }
        else if (A.row_major() && !B.row_major())
        {
          typedef viennacl::linalg::opencl::kernels::matrix_solve<NumericT, row_major, column_major>    KernelClass;
          KernelClass::init(ctx);
          program_name = KernelClass::program_name();
        }
        else if (!A.row_major() && B.row_major())
        {
          typedef viennacl::linalg::opencl::kernels::matrix_solve<NumericT, column_major, row_major>    KernelClass;
          KernelClass::init(ctx);
          program_name = KernelClass::program_name();
        }
        else
        {
          typedef viennacl::linalg::opencl::kernels::matrix_solve<NumericT, column_major, column_major>    KernelClass;
          KernelClass::init(ctx);
          program_name = KernelClass::program_name();
        }

        std::stringstream ss;
        if (trans_A) ss << "trans_";
        ss << SOLVERTAG::name();
        if (trans_B) ss << "_trans";
        ss << "_solve";

        viennacl::ocl::kernel & k = ctx.get_kernel(program_name, ss.str());

        if (trans_B)
          k.global_work_size(0, B.size1() * k.local_work_size());
        else
          k.global_work_size(0, B.size2() * k.local_work_size());
        detail::inplace_solve_impl(A, B, k);
      }



      //
      //  Solve on vector
      //

      template <typename NumericT, typename SOLVERTAG>
      void inplace_solve(const matrix_base<NumericT> & mat, bool trans_mat,
                               vector_base<NumericT> & vec,
                         SOLVERTAG)
      {
        cl_uint options = detail::get_option_for_solver_tag(SOLVERTAG());
        if (trans_mat)
          options |= 0x02;

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat,  "triangular_substitute_inplace");

        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),         cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),        cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),          cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)), cl_uint(viennacl::traits::internal_size2(mat)),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),
                                 options
                                )
                              );
      }

    }
  }
}

#endif
