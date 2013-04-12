#include <boost/python.hpp>

const char* hello() { return "Hi, I'm PyViennaCL!"; }

BOOST_PYTHON_MODULE(_viennacl)
{
  using namespace boost::python;

  def("hello", hello);
}

