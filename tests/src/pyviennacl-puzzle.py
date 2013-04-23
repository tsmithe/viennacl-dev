#!/usr/bin/env python3
"""
Compare the execution times of the two vector operations x = y1 + y2;
and x = y1 + y2 + y1 + y2; for ViennaCL vectors x, y1, and y2 for
different sizes (from about 100 entries to about 1.000.000
entries). Plot the curves and explain the differences you observe. You
are free to use either of the OpenMP, CUDA, or OpenCL backends.
"""

import os, sys
if sys.version_info.major < 3:
    print("You need to use Python 3 for this!")
    sys.exit(os.EX_CONFIG)

import numpy, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def run_test_add(v, max_size=2147483648, iterations=10000, per_entry=False):
    """
    Benchmarks vector addition using pyviennacl.

    v is the implementation of pyviennacl to use.
    max_size is the maximum vector size to benchmark.
    iterations is the number of times to perform each test.

    Executes (x = y1 + y2) and (x = y1 + y2 + y1 + y2) for vector
    sizes from 1 to max_size, with a logarithmic (base 2) step
    size. Runs each test `iterations' times, taking the mean.

    Prints the average execution times on each step.

    Returns a list of tuples (n, a, b) where n in the vector size, a is
    the average time taken for (y1+y2), and b is the time for (y1+y2+y1+y2)
    """

    bench = []
    n = 2;

    while n <= max_size:
        a = 0.0

        try:
            y1 = v.vector(n, 3.142)
            y2 = v.vector(n, 2.718)
        except:
            y1 = v.scalar_vector(n, 3.142)
            y2 = v.scalar_vector(n, 2.718)
        
        # Startup calculations
        x1 = y1+y2;
        x2 = y1+y2+y1+y2;
        try:
            print("\t\t\t\t\t\t%g" % x1.value[1])
            print("\t\t\t\t\t\t%g" % x2.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x1.get_value()[1])
            print("\t\t\t\t\t\t%g" % x2.get_value()[1])
        v.backend_finish()

        t1 = time.time()
        for m in range(iterations):
            x1 = y1 + y2
        v.backend_finish()
        t2 = time.time()
        a = t2 - t1
        try:
            print("\t\t\t\t\t\t%g" % x1.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x1.get_value()[1])
        v.backend_finish()

        t1 = time.time()
        for m in range(iterations):
            x2 = y1 + y2 + y1 + y2
        v.backend_finish()
        t2 = time.time()
        b = t2 - t1
        try:
            print("\t\t\t\t\t\t%g" % x2.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x2.get_value()[1])

        a /= iterations
        b /= iterations

        if per_entry:
            a /= n
            b /= n
        
        print("{:d}\t\t{:.6g}\t{:.6g}".format(n, a, b))
        bench.append((n, a, b))
        n *= 2
        
    return bench

def run_test_mul(v, max_size=2147483648, iterations=10000):
    """
    Benchmarks vector scaling using pyviennacl.

    v is the implementation of pyviennacl to use.
    max_size is the maximum vector size to benchmark.
    iterations is the number of times to perform each test.

    Executes (x1 = y1 * 2.0) for vector sizes from 1 to max_size, with
    a logarithmic (base 2) step size. Runs each test `iterations'
    times, taking the mean.

    Prints the average execution times on each step.

    Returns a list of tuples (n, a) where n in the vector size, a is
    the average time taken for (x1 = y1 * 2.0)
    """

    bench = []
    n = 2;

    while n <= max_size:
        a = 0.0

        try:
            y1 = v.vector(n, 3.142)
        except:
            y1 = v.scalar_vector(n, 3.142)
        
        # Startup calculations
        x1 = y1 * 2.0;
        try:
            print("\t\t\t\t\t\t%g" % x1.value[1])
            print("\t\t\t\t\t\t%g" % y1.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x1.get_value()[1])
            print("\t\t\t\t\t\t%g" % y1.get_value()[1])
        v.backend_finish()

        t1 = time.time()
        for m in range(iterations):
            x1 = y1 * 2.0;
        v.backend_finish()
        t2 = time.time()
        a = t2 - t1
        try:
            print("\t\t\t\t\t\t%g" % x1.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x1.get_value()[1])
        v.backend_finish()

        a /= iterations
        
        print("{:d}\t\t{:.6g}".format(n, a))
        bench.append((n, a))
        n *= 2
        
    return bench

def run_test_iadd(v, max_size=2147483648, iterations=10000):
    """
    Benchmarks in-place vector addition using pyviennacl.

    v is the implementation of pyviennacl to use.
    max_size is the maximum vector size to benchmark.
    iterations is the number of times to perform each test.

    Executes (x1 += y1) for vector sizes from 1 to max_size, with a
    logarithmic (base 2) step size. Runs each test `iterations' times,
    taking the mean.

    Prints the average execution times on each step.

    Returns a list of tuples (n, a) where n in the vector size, and a
    is the average time taken for (x1 += y1).
    """

    bench = []
    n = 2;

    while n <= max_size:
        a = 0.0
        b = 0.0

        try:
            x1 = v.vector(n, 3.142)
            y1 = v.vector(n, 2.718)
        except:
            x1 = v.scalar_vector(n, 3.142)
            y1 = v.scalar_vector(n, 2.718)
        
        # Startup calculations
        x1 += y1;
        try:
            print("\t\t\t\t\t\t%g" % x1.value[1])
            print("\t\t\t\t\t\t%g" % y1.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x1.get_value()[1])
            print("\t\t\t\t\t\t%g" % y1.get_value()[1])
        v.backend_finish()

        t1 = time.time()
        for m in range(iterations):
            x1 += y1
        v.backend_finish()
        t2 = time.time()
        a = t2 - t1
        try:
            print("\t\t\t\t\t\t%g" % x1.value[1])
        except:
            print("\t\t\t\t\t\t%g" % x1.get_value()[1])
        v.backend_finish()

        a /= iterations
        
        print("{:d}\t\t{:.6g}".format(n, a))
        bench.append((n, a))
        n *= 2
        
    return bench


def plot_test(bench, title=None, test=1, ylocs=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    """
    matplotlib-based graphing of benchmark results.
    test values:
     1 -> add
     2 -> add_assign
     3 -> times
    """
    x = [x[0] for x in bench]
    x1 = [x[1] for x in bench]
    if test == 1:
        x2 = [x[2] for x in bench]

    fig = plt.figure()
    sub = fig.add_subplot(111)
    if test == 1:
        sub.plot(x, x1, label="x1 = y1 + y2")
        sub.plot(x, x2, label="x2 = y1 + y2 + y1 + y2")
    elif test == 2:
        sub.plot(x, x1, label="x1 += y1")
    elif test == 3:
        sub.plot(x, x1, label="x1 = y1 * 2.0")
    elif test == 4:
        sub.plot(x, x1, label="copy(cpu, gpu)")
    else:
        raise Exception("Test index not recognised")
    plt.xlabel("Vector length")
    sub.set_xscale("log", basex=2)
    sub.set_yscale("log")
    #ylocs, ylabels = plt.yticks()
    if ylocs:
        plt.yticks(ylocs) #, ["%.1f" % y for y in ylocs*1e6])
    #plt.ylabel("Execution time (microseconds)")
    plt.ylabel("Execution time (seconds)")
    sub.legend(loc='best', fancybox=True)
    #sub.legend(loc=7, fancybox=True)
    sub.set_title(title)

    return fig

if __name__ == "__main__":
    max_size = 2**20
    iterations = 10**3
    figures = []
    fail = 0
    
#    try:
    print("Using pure C++...")
    from pyviennacl import _puzzle as p 
    print("Testing copy...")
    figures.append(plot_test(p.run_test_transfer(max_size, iterations), "Implementation (copy): _puzzle.cpp", 4))
    print("Testing +...")
    figures.append(plot_test(p.run_test_add(max_size, iterations), "Implementation (+): _puzzle.cpp", 1))
    print("Testing +=..")
    figures.append(plot_test(p.run_test_iadd(max_size, iterations), "Implementation (+=): _puzzle.cpp", 2))
    print("Testing *...")
    figures.append(plot_test(p.run_test_mul(max_size, iterations), "Implementation (*): _puzzle.cpp", 3))
#    except:
#        fail = 1
        
#    try:
    print("Using _viennacl....")
    from pyviennacl import _viennacl as v
    print("Testing +...")
    figures.append(plot_test(run_test_add(v, max_size, iterations), "Implementation (+): _viennacl.cpp", 1))
    print("Testing +=...")
    figures.append(plot_test(run_test_iadd(v, max_size, iterations), "Implementation (+=): _viennacl.cpp", 2))
    print("Testing *...")
    figures.append(plot_test(run_test_mul(v, max_size, iterations), "Implementation (*): _viennacl.cpp", 3))
#    except:
#        fail = 1
#    
#    try:
#        print("Using pyviennacl....")
#        import pyviennacl as v
#        figures.append(plot_test(run_test(v, max_size, iterations), "Implementation: pyviennacl"))
#    except:
#        fail = 1
        
    fname = "pyviennacl-puzzle.pdf"
    pp = PdfPages(fname)
    for fig in figures: fig.savefig(pp, format='pdf', bbox_inches=0)
    pp.close()

    print("\nSaved %d graphs to '%s'" % (len(figures), fname))

    if fail:
        sys.exit(os.EX_SOFTWARE)
    else:
        sys.exit(os.EX_OK)

