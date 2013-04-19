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

  std::cout << "Clock period is ";
  std::cout << (double) high_resolution_clock::period::num
                      / high_resolution_clock::period::den;
  std::cout << " seconds." << std::endl;

  // ViennaCL doesn't seem to work for vectors of length one
  for (n = 2; n<=max_size; n *= 2) {
    a = 0; b = 0;

    v x1(n);
    v x2(n);
    v y1(viennacl::scalar_vector<double>(n, 3.142));
    v y2(viennacl::scalar_vector<double>(n, 2.718));

    // Startup calculations
    x1 = y1+y2;
    x2 = y1+y2+y1+y2;
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x1(1)));
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x2(1)));
    viennacl::backend::finish();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      x1 = y1 + y2;
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    a = duration_cast<duration<double>>(t2 - t1).count();
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x1(1)));
    viennacl::backend::finish();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      x2 = y1 + y2 + y1 + y2;
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    b = duration_cast<duration<double>>(t2 - t1).count();
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x2(1)));

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

