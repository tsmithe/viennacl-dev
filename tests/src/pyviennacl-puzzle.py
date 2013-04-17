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

def run_test(v, max_size=2147483648, iterations=10000):
    """
    Benchmarks vector addition using pyviennacl.

    v is the implementation of pyviennacl to use.

    Executes (x = y1 + y2) and (x = y1 + y2 + y1 + y2) for vector sizes from 
    1 to 2*10^6, with a step of size=1000. Runs each test 100 times, taking
    the mean.

    Prints the average execution times on each step.

    Returns a list of tuples (n, a, b, c) where n in the vector size, a is
    the average time taken for (y1+y2), b is the time for (y1+y2+y1+y2), and
    c is the time for (y1+y2+y1+y2) given pure numpy types.
    """

    bench = []
    n = 1;

    while n <= max_size:
        a = 0.0
        b = 0.0

        try:
            y1 = v.vector(n*1000, 3.142)
            y2 = v.vector(n*1000, 2.718)
        except:
            y1 = v.scalar_vector(1000, 3.142)
            y2 = v.scalar_vector(1000, 2.718)
        
        # Startup calculations
        x1 = y1+y2;
        x2 = y1+y2+y1+y2;
        v.backend_finish()

        t1 = time.time()
        for m in range(iterations):
            x1 = y1 + y2
        v.backend_finish()
        t2 = time.time()
        a = t2 - t1

        t1 = time.time()
        for m in range(iterations):
            x2 = y1 + y2 + y1 + y2
        v.backend_finish()
        t2 = time.time()
        b = t2 - t1

        a /= iterations
        b /= iterations
        
        print("{:d}\t\t{:.6g}\t{:.6g}".format(n, a, b))
        bench.append((n, a, b))
        n *= 2
        
    return bench

def plot_test(bench, title=None):
    """
    matplotlib-based graphing of benchmark results.
    """
    x = [x[0] for x in bench]
    x1 = [x[1] for x in bench]
    x2 = [x[2] for x in bench]

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.plot(x, x1, label="x1 = y1 + y2")
    sub.plot(x, x2, label="x2 = y1 + y2 + y1 + y2")
    plt.xlabel("Vector length")
    ylocs, ylabels = plt.yticks()
    plt.yticks(ylocs, ["%.1f" % y for y in ylocs*1e6])
    plt.ylabel("Microseconds per addition")
    sub.set_xscale("log")
    #sub.set_yscale("log")
    #sub.legend(loc='best', fancybox=True)
    sub.legend(loc=7, fancybox=True)
    sub.set_title(title)

    return fig

if __name__ == "__main__":
    max_size = 2**21
    iterations = 10**4
    figures = []
    fail = 0
    
    try:
        print("Using pure C++...");
        from pyviennacl import _puzzle as p 
        figures.append(plot_test(p.run_test(max_size, iterations), "Implementation: _puzzle.cpp"))
    except:
        fail = 1
        
    try:
        print("Using _viennacl....")
        from pyviennacl import _viennacl as v
        figures.append(plot_test(run_test(v, max_size, iterations), "Implementation: _viennacl.cpp"))
    except:
        fail = 1
    
    try:
        print("Using pyviennacl....")
        import pyviennacl as v
        figures.append(plot_test(run_test(v, max_size, iterations), "Implementation: pyviennacl"))
    except:
        fail = 1
        
    fname = "pyviennacl-puzzle.pdf"
    pp = PdfPages(fname)
    for fig in figures: fig.savefig(pp, format='pdf', bbox_inches=0)
    pp.close()

    print("\nSaved %d graphs to '%s'" % (len(figures), fname))

    if fail:
        sys.exit(os.EX_SOFTWARE)
    else:
        sys.exit(os.EX_OK)

