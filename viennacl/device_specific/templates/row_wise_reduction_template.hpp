#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_ROW_WISE_REDUCTION_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_ROW_WISE_REDUCTION_HPP

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

#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing/read_write.hpp"
#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/templates/reduction_utils.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/scheduler/io.hpp"

namespace viennacl{

  namespace device_specific{

    class row_wise_reduction_template : public template_base{
    public:
      class parameters : public template_base::parameters
      {
      public:
        parameters(const char * _scalartype, char _A_trans, unsigned int _simd_width,
                   unsigned int _local_size_0, unsigned int _local_size_1,
                   unsigned int _num_groups_0): template_base::parameters(_scalartype, _simd_width, _local_size_0, _local_size_1, 1),
                                               num_groups_0(_num_groups_0), A_trans(_A_trans){ }


        unsigned int num_groups_0;
        char A_trans;
      };

    private:

      unsigned int lmem_used(unsigned int scalartype_size) const
      {
        return parameters_.local_size_0*(parameters_.local_size_1+1)*scalartype_size;
      }

      void configure_impl(vcl_size_t /*kernel_id*/, viennacl::ocl::context & /*context*/, statements_container const & statements, viennacl::ocl::kernel & kernel, unsigned int & n_arg)  const
      {
        kernel.global_work_size(0,parameters_.local_size_0*parameters_.num_groups_0);
        kernel.global_work_size(1,parameters_.local_size_1);

        scheduler::statement::container_type const & array = statements.data().begin()->array();
        vcl_size_t root = statements.data().begin()->root();
        scheduler::statement_node const * node = &array[array[root].rhs.node_index];
        bool trans = false;
        while(node->lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
        {
          if(node->op.type==scheduler::OPERATION_UNARY_TRANS_TYPE)
            trans = !trans;
          node = &array[node->lhs.node_index];
        }

        kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(node->lhs, utils::size1_fun())));
        kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(node->lhs, utils::size2_fun())));
      }

      void add_kernel_arguments(statements_container const & /*statements*/, std::string & arguments_string) const
      {
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }


      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        std::vector<mapped_vector_reduction*> exprs;
        for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
          for(mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit)
            if(mapped_vector_reduction * p = dynamic_cast<mapped_vector_reduction*>(iit->second.get()))
              exprs.push_back(p);
        std::size_t N = exprs.size();

        std::vector<scheduler::op_element> rops(N);
        std::vector<std::string> accs(N);
        std::vector<std::string> local_buffers_names(N);
        for(unsigned int k = 0 ; k < N ; ++k){
          scheduler::op_element root_op = exprs[k]->root_node().op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE){
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          }
          else{
            rops[k].type        = root_op.type;
          }
          accs[k] = "sum"+tools::to_string(k);
          local_buffers_names[k] = "buf"+tools::to_string(k);
        }



        unsigned int lsize0 = parameters_.local_size_0;
        unsigned int lsize1 = parameters_.local_size_1+1;

        std::string size1 = "M";
        std::string size2 = "N";
        if(parameters_.A_trans=='T')
          std::swap(size1, size2);

        for(std::vector<mapped_vector_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
          stream << "__local " <<  (*it)->scalartype() << " buf" << std::distance(exprs.begin(), it) << '[' << lsize0*lsize1 << "];" << std::endl;

        stream << "unsigned int lid0 = get_local_id(0);" << std::endl;
        stream << "unsigned int lid1 = get_local_id(1);" << std::endl;


        stream << "for(unsigned int r = get_global_id(0) ; r < " << size1 << " ; r += get_global_size(0)){" << std::endl;
        stream.inc_tab();
        {
          for(unsigned int k = 0 ; k < exprs.size() ; ++k)
            stream << exprs[k]->scalartype() << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;

          stream << "for( unsigned int c = get_local_id(1) ; c < " << size2 << " ; c += get_local_size(1)){" << std::endl;
          stream.inc_tab();
          {
            std::set<std::string>  cache;
            std::size_t N = exprs.size();

            for(unsigned int k = 0 ; k < N ; ++k)
            {
              viennacl::scheduler::statement const & statement = exprs[k]->statement();
              viennacl::scheduler::statement_node const & root_node = exprs[k]->root_node();
              tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, parameters_.simd_width, "reg", cache, statement, exprs[k]->root_idx(), index_tuple("r", size1, "c", size2),stream,exprs[k]->mapping(), tree_parsing::LHS_NODE_TYPE);

              if(root_node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE)
                tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, parameters_.simd_width, "reg", cache, statement, exprs[k]->root_idx(), index_tuple("c", size1), stream,exprs[k]->mapping(), tree_parsing::RHS_NODE_TYPE);
            }


            //Update sums;
            for(unsigned int k = 0 ; k < N ; ++k)
            {
              viennacl::scheduler::statement const & statement = exprs[k]->statement();
              vcl_size_t root_idx = exprs[k]->root_idx();
              scheduler::statement_node const & root_node = exprs[k]->root_node();
              std::string value = tree_parsing::evaluate_expression(statement, root_idx, index_tuple("",""), 0, exprs[k]->mapping(), tree_parsing::LHS_NODE_TYPE);
              if(root_node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE)
              {
                value += "*";
                value += tree_parsing::evaluate_expression(statement, root_idx, index_tuple("",""), 0, exprs[k]->mapping(), tree_parsing::RHS_NODE_TYPE);
              }
              compute_reduction(stream,"","",accs[k],value,rops[k]);
            }
          }
          stream.dec_tab();
          stream << "}" << std::endl;


          for(unsigned int k = 0 ; k < exprs.size() ; ++k){
            stream << "buf" << k << "[lid0*" << lsize1 << "+ lid1] = " << accs[k] << ";" << std::endl;
          }

          for(unsigned int stride = parameters_.local_size_1/2 ; stride>0 ; stride /=2){
            stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
            stream <<  "if(lid1 < " << stride << ")" ;
            stream << "{" << std::endl;
            stream.inc_tab();

            for(unsigned int k = 0 ; k < N ; ++k)
              compute_reduction(stream, "", ""
                                    ,local_buffers_names[k] + "[lid0*" + tools::to_string(lsize1) + "+ lid1]"
                                    ,local_buffers_names[k] + "[lid0*" + tools::to_string(lsize1) + "+ lid1 + " + tools::to_string(stride) + "]"
                                    ,rops[k]);

            stream.dec_tab();
            stream << "}" << std::endl;
          }


          stream <<  "if(lid1 == 0)" ;
          stream << "{" << std::endl;
          stream.inc_tab();
          for(unsigned int k = 0 ; k < N ; ++k)
            exprs[k]->access_name(local_buffers_names[k] + "[lid0*"+tools::to_string(lsize1)+"]");

          unsigned int i = 0;
          for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it){
            std::string str;
            tree_parsing::traverse(*it, it->root(), tree_parsing::evaluate_expression_traversal(index_tuple("r",size1, "0", size2), 0, str, mapping[i++]), false);
            stream << str << ";" << std::endl;
          }
          stream.dec_tab();
          stream << "}" << std::endl;

        }
        stream.dec_tab();
        stream << "}" << std::endl;
      }

    public:
      row_wise_reduction_template(row_wise_reduction_template::parameters const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(parameters, binding_policy), parameters_(parameters){ }

    private:
      row_wise_reduction_template::parameters const & parameters_;
    };

  }
}

#endif
