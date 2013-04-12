#include <boost/python.hpp>

const char* hello() { return "Hi, I'm ViennaCL's numpy!"; }

BOOST_PYTHON_MODULE(_viennacl_numpy)
{
  using namespace boost::python;

  def("hello", hello);
}

