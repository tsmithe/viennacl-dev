#include <boost/python.hpp>
#define VIENNACL_WITH_OPENCL
#include <viennacl/vector.hpp>
#include <chrono>
#include <cstdio>

#include <typeinfo>

using namespace boost::python;
using namespace std::chrono;

list run_test(unsigned int max_size, unsigned int iterations)
{

  typedef viennacl::vector<double> v;

  list bench;
  unsigned int n, m;
  high_resolution_clock::time_point t1, t2;
  double a, b;
  v x1, x2, y1, y2, y3, y4;

  std::cout << "Clock period is ";
  std::cout << (double) high_resolution_clock::period::num
                      / high_resolution_clock::period::den;
  std::cout << " seconds." << std::endl;

  for (n = 1; n<=max_size; n *= 2) {
    a = 0; b = 0;

    y1 = v(viennacl::scalar_vector<double>(n, 3.142));
    y2 = v(viennacl::scalar_vector<double>(n, 2.718));

    // Startup calculations
    x1 = y1+y2;
    x2 = y1+y2+y1+y2;
    viennacl::backend::finish();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      x1 = y1 + y2;
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    a = duration_cast<duration<double>>(t2 - t1).count();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      x2 = y1 + y2 + y1 + y2;
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    b = duration_cast<duration<double>>(t2 - t1).count();

    a /= iterations;
    b /= iterations;

    printf("%d\t\t%g\t%g\n", n, a, b);
    bench.append(make_tuple<unsigned int, double, double>(n, a, b));

  }
 
  return bench;

}

BOOST_PYTHON_MODULE(_puzzle)
{

  def("run_test", run_test);

}

