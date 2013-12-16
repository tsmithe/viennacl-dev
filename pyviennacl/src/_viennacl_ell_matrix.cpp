#include "_viennacl.h"

void export_ell_matrix() {
  bp::class_<vcl::ell_matrix<double>, 
             boost::shared_ptr<vcl::ell_matrix<double> >,
             boost::noncopyable >
    ("ell_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::ell_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::ell_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("nnz",
		  make_function(&vcl::ell_matrix<double>::nnz,
			      bp::return_value_policy<bp::return_by_value>()))

    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::ell_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)
    ;
}

