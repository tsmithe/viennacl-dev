#include "sparse_matrix.h"

PYVCL_SUBMODULE(hyb_matrix)
{
  bp::class_<vcl::hyb_matrix<double>, 
             vcl::tools::shared_ptr<vcl::hyb_matrix<double> >,
             boost::noncopyable >
    ("hyb_matrix", bp::no_init)
    .add_property("size1",
		  make_function(&vcl::hyb_matrix<double>::size1,
			      bp::return_value_policy<bp::return_by_value>()))

    .add_property("size2",
		  make_function(&vcl::hyb_matrix<double>::size2,
			      bp::return_value_policy<bp::return_by_value>()))

    .def("prod", pyvcl_do_2ary_op<vcl::vector<double>,
	 vcl::hyb_matrix<double>&, vcl::vector<double>&,
	 op_prod, 0>)
    ;
}

