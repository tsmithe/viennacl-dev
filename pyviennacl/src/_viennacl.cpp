#include <boost/python.hpp>
#define VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_PYTHON
#include <viennacl/vector.hpp>
#include <iostream>
#include <typeinfo>

using namespace viennacl;
using namespace boost::python;

typedef double scalar_type;

typedef std::vector<scalar_type> cpu_vector_type;
typedef vector<scalar_type, 1>   vcl_vector_type;

typedef vector_expression<const vcl_vector_type, const vcl_vector_type, op_add> vcl_vector_add_type;

/** @brief Returns a Python list describing the VCL_T */
// Need to reimplement using numpy.ndarray..
list vcl_vector_to_list(vcl_vector_type const& v)
{
  list l;
  cpu_vector_type c(v.size());
  viennacl::copy(v.begin(), v.end(), c.begin());

  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((vcl_vector_type::value_type::value_type)c[i]);
  
  return l;
}

template <class E>
list vcl_vector_expression_to_list(E const& v)
{
 
  // Highly dubious code...

  list l;

  typename E::VectorType r(v);
  cpu_vector_type c(v.size());
  viennacl::copy(r.begin(), r.end(), c.begin());

  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((typename E::VectorType::value_type::value_type)c[i]);
  
  return l;
}

template <class T, class L, class R>
T vcl_add(L a, R b)
{
  return a + b;
}

vcl_vector_type new_scalar_vector(int length, double value) {
  return static_cast<vcl_vector_type>(scalar_vector<scalar_type>(length, value));
}

BOOST_PYTHON_MODULE(_viennacl)
{

  class_<vcl_vector_add_type>("vector_expression", no_init)
    .def("__add__", vcl_add<vcl_vector_type, vcl_vector_add_type const&, vcl_vector_type const&>)
    .def("get_value", &vcl_vector_expression_to_list<vcl_vector_add_type>)
    ;

  class_<vcl_vector_type>("vector")
    .def(init<int>())
    .def(init<list>())
    .def(init<vcl_vector_type>())
    .def("__add__", vcl_add<vcl_vector_type, vcl_vector_type const&, vcl_vector_type const&>)
    .def("get_value", &vcl_vector_to_list)
    ;

  def("scalar_vector", new_scalar_vector);

}

