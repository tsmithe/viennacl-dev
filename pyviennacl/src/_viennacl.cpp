#include <boost/python.hpp>
#include <boost/numpy.hpp>
#define VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_PYTHON
#include <viennacl/vector.hpp>
#include <cstdint>
#include <iostream>
#include <typeinfo>

using namespace viennacl;
using namespace boost::python;

namespace np = boost::numpy;

typedef scalar<double> vcl_scalar_t;
typedef double         cpu_scalar_t;

typedef std::vector<cpu_scalar_t> cpu_vector_t;
typedef vector<cpu_scalar_t, 1>   vcl_vector_t;

template <class T, class L, class R> T vcl_add(L a, R b) { return a + b; }
template <class T, class L, class R> T vcl_iadd(L a, R b) { a += b; return a; }

template <class T, class R> T vcl_mul_obj_l(object const& a, R b) { return static_cast<double>(extract<double>(a)) * b; }
template <class T, class L> T vcl_mul_obj_r(L a, object const& b) { return a * static_cast<double>(extract<double>(b)); }


/** @brief Returns a Python list describing the VCL_T */
list vcl_vector_to_list(vcl_vector_t const& v)
{
  list l;
  cpu_vector_t c(v.size());
  copy(v.begin(), v.end(), c.begin());

  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((vcl_vector_t::value_type::value_type)c[i]);
  
  return l;
}

np::ndarray vcl_vector_to_ndarray(vcl_vector_t const& v)
{
  return np::from_object(vcl_vector_to_list(v), np::dtype::get_builtin<cpu_scalar_t>());
}

/** @brief Creates the vector from the supplied ndarray */
vcl_vector_t vector_from_ndarray(np::ndarray const& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    boost::python::throw_error_already_set();
  }
  
  uint32_t s = (uint32_t) array.shape(0);
  
  vcl_vector_t v(s);
  cpu_vector_t cpu_vector(s);
  
  for (uint32_t i=0; i < s; ++i)
    cpu_vector[i] = extract<cpu_scalar_t>(array[i]);
  
  fast_copy(cpu_vector.begin(), cpu_vector.end(), v.begin());

  return v;
}

/** @brief Creates the vector from the supplied Python list */
vcl_vector_t vector_from_list(boost::python::list const& l)
{
  return vector_from_ndarray(np::from_object(l, np::dtype::get_builtin<cpu_scalar_t>()));
}

vcl_vector_t new_scalar_vector(int length, double value) {
  return static_cast<vcl_vector_t>(scalar_vector<cpu_scalar_t>(length, value));
}

BOOST_PYTHON_MODULE(_viennacl)
{

  np::initialize();

  class_<vcl_vector_t>("vector")
    .def(init<int>())
    .def(init<vcl_vector_t>())
    .def("__add__", vcl_add<vcl_vector_t, vcl_vector_t const&, vcl_vector_t const&>)
    .def("__mul__", vcl_mul_obj_l<vcl_vector_t, vcl_vector_t const&>)
    .def("__mul__", vcl_mul_obj_r<vcl_vector_t, vcl_vector_t const&>)
    .def("__iadd__", vcl_iadd<vcl_vector_t, vcl_vector_t, vcl_vector_t const&>)
    .def("get_value", &vcl_vector_to_ndarray)
    ;

  def("vector_from_list", vector_from_list);
  def("vector_from_ndarray", vector_from_ndarray);
  def("scalar_vector", new_scalar_vector);

  def("backend_finish", backend::finish);

}

