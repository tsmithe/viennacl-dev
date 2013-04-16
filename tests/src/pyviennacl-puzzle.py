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

import matplotlib, numpy, time

def run_test(v):
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
    for n in range(1,2001):
        a = 0.0
        b = 0.0

        for m in range(100):
            try:
                y1 = v.vector(n*1000, 3.142)
                y2 = v.vector(n*1000, 2.718)
            except:
                y1 = v.scalar_vector(1000, 3.142)
                y2 = v.scalar_vector(1000, 2.718)

            t1 = time.time()
            x1 = y1 + y2
            t2 = time.time()
            a += t2 - t1

            t1 = time.time()
            x2 = y1 + y2 + y1 + y2
            t2 = time.time()
            b += t2 - t1

        a /= 100.0
        b /= 100.0
        
        print("{:d}\t\t{:.6g}\t{:.6g}".format(n*1000, a, b))
        
        bench.append((n*1000, a, b))
        
    return bench

def plot_test(bench):
    """
    Stub for matplotlib-based graphing of benchmark results.
    """
    return

if __name__ == "__main__":
#    print("Using pure C++...");
#    from pyviennacl import _puzzle as p
#    p.run_test()

    print("Using pyviennacl....")
    import pyviennacl as v
    run_test(v)

#    print("Using _viennacl....")
#    from pyviennacl import _viennacl as v
#    run_test(v)

    sys.exit(os.EX_OK)

