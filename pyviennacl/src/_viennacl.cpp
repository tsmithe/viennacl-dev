#include <boost/python.hpp>
#include <boost/numpy.hpp>
#define VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_PYTHON
#include <viennacl/vector.hpp>
#include <iostream>
#include <typeinfo>

using namespace viennacl;
using namespace boost::python;

namespace np = boost::numpy;

typedef double scalar_type;
typedef unsigned int size_type;

typedef std::vector<scalar_type> cpu_vector_type;
typedef vector<scalar_type, 1>   vcl_vector_type;

typedef vector_expression<const vcl_vector_type, const vcl_vector_type, op_add> vcl_vector_add_type;

template <class T, class L, class R> T vcl_add(L a, R b) { return a + b; }

/** @brief Returns a Python list describing the VCL_T */
list vcl_vector_to_list(vcl_vector_type const& v)
{
  list l;
  cpu_vector_type c(v.size());
  copy(v.begin(), v.end(), c.begin());

  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((vcl_vector_type::value_type::value_type)c[i]);
  
  return l;
}

np::ndarray vcl_vector_to_ndarray(vcl_vector_type const& v)
{
  return np::from_object(vcl_vector_to_list(v), np::dtype::get_builtin<scalar_type>());
}

template <class E>
list vcl_vector_expression_to_list(E const& v)
{
 
  // Highly dubious code...

  list l;

  typename E::VectorType r(v);
  cpu_vector_type c(v.size());
  copy(r.begin(), r.end(), c.begin());

  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((typename E::VectorType::value_type::value_type)c[i]);
  
  return l;
}

template <class E>
np::ndarray vcl_vector_expression_to_ndarray(E const& v)
{
  // Calls highly dubious code...
  return np::from_object(vcl_vector_expression_to_list<E>(v), np::dtype::get_builtin<scalar_type>());
}

/** @brief Creates the vector from the supplied ndarray */
vcl_vector_type vector_from_ndarray(np::ndarray const& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    boost::python::throw_error_already_set();
  }
  
  size_type s = (size_type) array.shape(0);
  
  vcl_vector_type v(s);
  cpu_vector_type cpu_vector(s);
  
  for (size_type i=0; i < s; ++i)
    cpu_vector[i] = extract<scalar_type>(array[i]);
  
  fast_copy(cpu_vector.begin(), cpu_vector.end(), v.begin());

  return v;
}

/** @brief Creates the vector from the supplied Python list */
vcl_vector_type vector_from_list(boost::python::list const& l)
{
  return vector_from_ndarray(np::from_object(l, np::dtype::get_builtin<scalar_type>()));
}

vcl_vector_type new_scalar_vector(int length, double value) {
  return static_cast<vcl_vector_type>(scalar_vector<scalar_type>(length, value));
}

BOOST_PYTHON_MODULE(_viennacl)
{

  np::initialize();

  class_<vcl_vector_add_type>("vector_expression", no_init)
    .def("__add__", vcl_add<vcl_vector_type, vcl_vector_add_type const&, vcl_vector_type const&>)
    .def("get_value", &vcl_vector_expression_to_ndarray<vcl_vector_add_type>)
    ;

  class_<vcl_vector_type>("vector")
    .def(init<int>())
     .def(init<vcl_vector_type>())
    .def("__add__", vcl_add<vcl_vector_type, vcl_vector_type const&, vcl_vector_type const&>)
    .def("get_value", &vcl_vector_to_ndarray)
    ;

  def("vector_from_list", vector_from_list);
  def("vector_from_ndarray", vector_from_ndarray);
  def("scalar_vector", new_scalar_vector);

}

