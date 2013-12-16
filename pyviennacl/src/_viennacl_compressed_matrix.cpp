#include "_viennacl.h"

void export_compressed_matrix() {
    bp::class_<vcl::compressed_matrix<double>,
             boost::shared_ptr<vcl::compressed_matrix<double> > >
    ("compressed_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::compressed_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::compressed_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("nnz",
		  make_function(&vcl::compressed_matrix<double>::nnz,
			      bp::return_value_policy<bp::return_by_value>()))

    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)
    
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<double>,
	 vcl::compressed_matrix<double>&, vcl::vector<double>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve, 0>)
    ;
}

