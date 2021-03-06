#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_UTILS_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_UTILS_HPP

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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


/** @file viennacl/generator/row_wise_reduction.hpp
 *
 * Kernel template for the vector reduction operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/device_specific/utils.hpp"


namespace viennacl{

  namespace device_specific{

    static void compute_reduction(utils::kernel_generation_stream & os, std::string accidx, std::string curidx, std::string const & acc, std::string const & cur, scheduler::op_element const & op){
        if(utils::is_index_reduction(op))
        {
          os << accidx << "= select(" << accidx << "," << curidx << "," << cur << ">" << acc << ");" << std::endl;
          os << acc << "=";
          if(op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE) os << "fmax";
          if(op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE) os << "max";
          if(op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE) os << "fmin";
          if(op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE) os << "min";
          os << "(" << acc << "," << cur << ");"<< std::endl;
        }
        else{
          os << acc << "=";
          if(utils::elementwise_function(op))
              os << tree_parsing::evaluate(op.type) << "(" << acc << "," << cur << ")";
          else
              os << "(" << acc << ")" << tree_parsing::evaluate(op.type)  << "(" << cur << ")";
          os << ";" << std::endl;
        }
    }


    inline void reduce_1d_local_memory(utils::kernel_generation_stream & stream, unsigned int size, std::vector<std::string> const & bufs, std::vector<scheduler::op_element> const & rops)
    {
        //Reduce local memory
        stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int stride = " << size/2 << "; stride >0 ; stride /=2){" << std::endl;
        stream.inc_tab();
        stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
        stream << "if(lid <  stride){" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < bufs.size() ; ++k){
            std::string acc = bufs[k] + "[lid]";
            std::string accidx = (bufs[k] + "idx") + "[lid]";
            std::string cur = bufs[k] + "[lid + stride]";
            std::string curidx = (bufs[k] + "idx") + "[lid + stride]";
            compute_reduction(stream,accidx,curidx,acc,cur,rops[k]);
        }
        stream.dec_tab();
        stream << "}" << std::endl;
        stream.dec_tab();
        stream << "}" << std::endl;
    }

    inline std::string neutral_element(scheduler::op_element const & op){
      switch(op.type){
        case scheduler::OPERATION_BINARY_ADD_TYPE : return "0";
        case scheduler::OPERATION_BINARY_MULT_TYPE : return "1";
        case scheduler::OPERATION_BINARY_DIV_TYPE : return "1";
        case scheduler::OPERATION_BINARY_ELEMENT_FMAX_TYPE : return "-INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE : return "-INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_MAX_TYPE : return "-INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE : return "-INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_FMIN_TYPE : return "INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE : return "INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_MIN_TYPE : return "INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE : return "INFINITY";

        default: throw generator_not_supported_exception("Unsupported reduction operator : no neutral element known");
      }
    }

  }
}

#endif
