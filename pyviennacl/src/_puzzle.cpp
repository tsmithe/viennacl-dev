#include <boost/python.hpp>
#define VIENNACL_WITH_OPENCL
#include <viennacl/vector.hpp>
#include <chrono>
#include <cstdio>

#include <typeinfo>

using namespace boost::python;
using namespace std::chrono;

list run_test_add(unsigned int max_size, unsigned int iterations)
{

  // x = y + x

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

list run_test_iadd(unsigned int max_size, unsigned int iterations)
{

  // x += y

  typedef viennacl::vector<double> v;

  list bench;
  unsigned int n, m;
  high_resolution_clock::time_point t1, t2;
  double a;

  std::cout << "Clock period is ";
  std::cout << (double) high_resolution_clock::period::num
                      / high_resolution_clock::period::den;
  std::cout << " seconds." << std::endl;

  // ViennaCL doesn't seem to work for vectors of length one
  for (n = 2; n<=max_size; n *= 2) {
    a = 0;

    v x1(viennacl::scalar_vector<double>(n, 3.142));
    v y1(viennacl::scalar_vector<double>(n, 2.718));

    // Startup calculations
    x1 += y1;
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x1(1)));
    viennacl::backend::finish();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      x1 += y1;
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    a = duration_cast<duration<double>>(t2 - t1).count();
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x1(1)));
    viennacl::backend::finish();

    a /= iterations;

    printf("%d\t\t%g\n", n, a);
    bench.append(make_tuple<unsigned int, double>(n, a));
  }
 
  return bench;

}

list run_test_mul(unsigned int max_size, unsigned int iterations)
{

  // x = 2 * y

  typedef viennacl::vector<double> v;

  list bench;
  unsigned int n, m;
  high_resolution_clock::time_point t1, t2;
  double a;

  std::cout << "Clock period is ";
  std::cout << (double) high_resolution_clock::period::num
                      / high_resolution_clock::period::den;
  std::cout << " seconds." << std::endl;

  // ViennaCL doesn't seem to work for vectors of length one
  for (n = 2; n<=max_size; n *= 2) {
    a = 0;

    v x1(n);
    v y1(viennacl::scalar_vector<double>(n, 3.142));

    // Startup calculations
    x1 = 2.0 * y1;
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x1(1)));
    viennacl::backend::finish();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      x1 = 2.0 * y1;
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    a = duration_cast<duration<double>>(t2 - t1).count();
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(x1(1)));
    viennacl::backend::finish();

    a /= iterations;

    printf("%d\t\t%g\n", n, a);
    bench.append(make_tuple<unsigned int, double>(n, a));
  }
 
  return bench;

}

list run_test_transfer(unsigned int max_size, unsigned int iterations)
{

  // simple vector transfer benchmark

  typedef viennacl::vector<double> v;

  list bench;
  unsigned int n, m;
  high_resolution_clock::time_point t1, t2;
  double a;

  std::cout << "Clock period is ";
  std::cout << (double) high_resolution_clock::period::num
                      / high_resolution_clock::period::den;
  std::cout << " seconds." << std::endl;

  // ViennaCL doesn't seem to work for vectors of length one
  for (n = 2; n<=max_size; n *= 2) {
    a = 0;

    std::vector<double> cpu_vector(n);
    v gpu_vector(n);

    for (m = 0; m<n; ++m)
      cpu_vector[m] = (double) 3.142;
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(cpu_vector[1]));

    // Startup calculations
    viennacl::fast_copy(cpu_vector.begin(), cpu_vector.end(), gpu_vector.begin());
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(gpu_vector[1]));
    viennacl::backend::finish();

    t1 = high_resolution_clock::now();
    for (m = 0; m<iterations; ++m)
      viennacl::fast_copy(cpu_vector.begin(), cpu_vector.end(), gpu_vector.begin());
    viennacl::backend::finish();
    t2 = high_resolution_clock::now();
    a = duration_cast<duration<double>>(t2 - t1).count();
    printf("\t\t\t\t\t\t%g\n", static_cast<double>(gpu_vector[1]));
    viennacl::backend::finish();

    a /= iterations;

    printf("%d\t\t%g\n", n, a);
    bench.append(make_tuple<unsigned int, double>(n, a));
  }
 
  return bench;

}


BOOST_PYTHON_MODULE(_puzzle)
{

  def("run_test_add", run_test_add);
  def("run_test_iadd", run_test_iadd);
  def("run_test_mul", run_test_mul);

  def("run_test_transfer", run_test_transfer);

}

