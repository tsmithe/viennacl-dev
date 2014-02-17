#include "sparse_matrix.h"

PYVCL_SUBMODULE(compressed_matrix)
{
  // TODO: Other types than double!
  bp::class_<cpu_compressed_matrix_wrapper<double> >
    ("cpu_compressed_matrix_double")
    .def(bp::init<>())
    .def(bp::init<uint32_t, uint32_t>())
    .def(bp::init<uint32_t, uint32_t, uint32_t>())
    .def(bp::init<cpu_compressed_matrix_wrapper<double> >())
    .def(bp::init<vcl::compressed_matrix<double> >())
    //.def(bp::init<vcl::coordinate_matrix<double> >())
    .def(bp::init<vcl::ell_matrix<double> >())
    .def(bp::init<vcl::hyb_matrix<double> >())
    .def(bp::init<np::ndarray>())
    .add_property("nonzeros", &cpu_compressed_matrix_wrapper<double>::places_to_python)
    .add_property("nnz", &cpu_compressed_matrix_wrapper<double>::nnz)
    .add_property("size1", &cpu_compressed_matrix_wrapper<double>::size1)
    .add_property("size2", &cpu_compressed_matrix_wrapper<double>::size2)
    .def("print_places", &cpu_compressed_matrix_wrapper<double>::print_places)
    .def("resize", &cpu_compressed_matrix_wrapper<double>::resize)
    .def("set_entry", &cpu_compressed_matrix_wrapper<double>::set_entry)
    .def("get_entry", &cpu_compressed_matrix_wrapper<double>::get_entry)
    .def("as_ndarray", &cpu_compressed_matrix_wrapper<double>::as_ndarray)
    .def("as_compressed_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix_with_size<vcl::compressed_matrix<double> >)
    .def("as_coordinate_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix_with_size<vcl::coordinate_matrix<double> >)
    .def("as_ell_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix<vcl::ell_matrix<double> >)
    .def("as_hyb_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix<vcl::hyb_matrix<double> >)
    ;

    bp::class_<vcl::compressed_matrix<double>,
             vcl::tools::shared_ptr<vcl::compressed_matrix<double> > >
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

      /*    
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
      */
    ;
}

