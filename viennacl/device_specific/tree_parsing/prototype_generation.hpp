#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_PROTOTYPE_GENERATION_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_PROTOTYPE_GENERATION_HPP

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


/** @file viennacl/generator/set_arguments_functor.hpp
    @brief Functor to set the arguments of a statement into a kernel
*/

#include <set>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"

#include "viennacl/device_specific/tree_parsing/traverse.hpp"



namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      /** @brief functor for generating the prototype of a statement */
      class prototype_generation_traversal : public traversal_functor{
        private:
          unsigned int simd_width_;
          std::set<std::string> & already_generated_;
          std::string & str_;
          mapping_type const & mapping_;
        public:
          prototype_generation_traversal(unsigned int simd_width, std::set<std::string> & already_generated, std::string & str, mapping_type const & mapping) : simd_width_(simd_width), already_generated_(already_generated), str_(str),  mapping_(mapping){ }

          void operator()(scheduler::statement const & statement, vcl_size_t root_idx, node_type node_type) const {
              scheduler::statement_node const & root_node = statement.array()[root_idx];
              if( (node_type==LHS_NODE_TYPE && root_node.lhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY)
                ||(node_type==RHS_NODE_TYPE && root_node.rhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY) )
                  mapping_.at(std::make_pair(root_idx,node_type))->append_kernel_arguments(simd_width_, already_generated_, str_);
          }
      };

    }

  }

}
#endif
