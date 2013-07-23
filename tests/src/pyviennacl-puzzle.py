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

def run_test_add(v, max_size=1048577, iterations=10000,
                 per_entry=False, scheduler=False):
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

    v.backend_finish()

    while n <= max_size:
        a = 0.0

        y1 = v.vector(n, 3.142)
        y2 = v.vector(n, 2.718)
        
        if scheduler:
            x1 = y1+y2;
            x2 = v.vector(n)

            node1 = v.statement_node(
                v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
                v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE,
                v.statement_node_type_family.VECTOR_TYPE_FAMILY,
                v.statement_node_type.VECTOR_DOUBLE_TYPE,
                v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY,
                v.statement_node_type.COMPOSITE_OPERATION_TYPE)
            node1.set_lhs_vector_double(x2)
            node1.set_rhs_node_index(1)
            
            node2 = v.statement_node(
                v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
                v.operation_node_type.OPERATION_BINARY_ADD_TYPE,
                v.statement_node_type_family.VECTOR_TYPE_FAMILY,
                v.statement_node_type.VECTOR_DOUBLE_TYPE,
                v.statement_node_type_family.VECTOR_TYPE_FAMILY,
                v.statement_node_type.VECTOR_DOUBLE_TYPE)
            node2.set_lhs_vector_double(y1)
            node2.set_rhs_vector_double(y2)

            s = v.statement()
            s.insert_at_begin(node2)
            s.insert_at_begin(node1)

            # Startup calculation
            s.execute()

        else:
            # Startup calculations
            x1 = y1+y2;
            x2 = y1+y2+y1+y2;

        print(n)
        
        print("\t\t\t\t\t\t%g" % x1.as_ndarray()[1])
        print("\t\t\t\t\t\t%g" % x2.as_ndarray()[1])
        v.backend_finish()

        t1 = time.time()
        for m in range(iterations):
            x1 = y1 + y2
            v.backend_finish()
        t2 = time.time()
        a = t2 - t1
        print("\t\t\t\t\t\t%g" % x1.as_ndarray()[1])
        v.backend_finish()

        if scheduler:
            t1 = time.time()
            for m in range(iterations):
                node1 = v.statement_node(
                    v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
                    v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE,
                    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
                    v.statement_node_type.VECTOR_DOUBLE_TYPE,
                    v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY,
                    v.statement_node_type.COMPOSITE_OPERATION_TYPE)
                node1.set_lhs_vector_double(x2)
                node1.set_rhs_node_index(1)
                
                node2 = v.statement_node(
                    v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
                    v.operation_node_type.OPERATION_BINARY_ADD_TYPE,
                    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
                    v.statement_node_type.VECTOR_DOUBLE_TYPE,
                    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
                    v.statement_node_type.VECTOR_DOUBLE_TYPE)
                node2.set_lhs_vector_double(y1)
                node2.set_rhs_vector_double(y2)
                
                s = v.statement()
                s.insert_at_begin(node2)
                s.insert_at_begin(node1)

                s.execute()
                v.backend_finish()
            t2 = time.time()
            b = t2 - t1
            pass
        else:
            t1 = time.time()
            for m in range(iterations):
                x2 = y1 + y2 + y1 + y2
                v.backend_finish()
            t2 = time.time()
            b = t2 - t1
        print("\t\t\t\t\t\t%g" % x2.as_ndarray()[1])

        a /= iterations
        b /= iterations

        if per_entry:
            a /= n
            b /= n
        
        print("{:d}\t\t{:.6g}\t{:.6g}".format(n, a, b))
        bench.append((n, a, b))
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
    max_size = 2**17
    iterations = 10**3
    figures = []
    fail = 0
    
#    try:
    #print("Using pure C++...")
    #from pyviennacl import _puzzle as p 
    #print("Testing copy...")
    #figures.append(plot_test(p.run_test_transfer(max_size, iterations), "Implementation (copy): _puzzle.cpp", 4))
    #print("Testing +...")
    #figures.append(plot_test(p.run_test_add(max_size, iterations), "Implementation (+): _puzzle.cpp", 1))
    #del p
#    except:
#        fail = 1
        
#    try:
    print("Using _viennacl....")
    from pyviennacl import _viennacl as v
    print("Testing +...")
    run_test_add(v, max_size, iterations, scheduler=False)
    figures.append(plot_test(run_test_add(v, max_size, iterations, scheduler=False), "Implementation (+): _viennacl.cpp", 1))
    print("Testing +...")
    #run_test_add(v, max_size, iterations, scheduler=True)
    figures.append(plot_test(run_test_add(v, max_size, iterations, scheduler=True), "Implementation (+): _viennacl.cpp", 1))
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

