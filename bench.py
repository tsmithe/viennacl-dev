import numpy as np
import pyviennacl as p
import pyviennacl._viennacl as _v
import time

def dobench2(size1, size2, dtype=np.float64):
    u = p.Matrix(size1, size2, 0.1, dtype=dtype)
    v = p.Matrix(size1, size2, 0.2, dtype=dtype)
    p.Statement(u+v).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        p.Statement(u+v).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench4(size1, size2, dtype=np.float64):
    u = p.Matrix(size1, size2, 0.1, dtype=dtype)
    v = p.Matrix(size1, size2, 0.2, dtype=dtype)
    w = p.Matrix(size1, size2, 0.3, dtype=dtype)
    x = p.Matrix(size1, size2, 0.4, dtype=dtype)
    p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_vcl2(size1, size2):
    u = _v.matrix(size1, size2, 0.1)
    v = _v.matrix(size1, size2, 0.2)
    y = u+v
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        y = u+v
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_vcl4(size1, size2):
    u = _v.matrix(size1, size2, 0.1)
    v = _v.matrix(size1, size2, 0.2)
    w = _v.matrix(size1, size2, 0.3)
    x = _v.matrix(size1, size2, 0.4)
    y = u+v+w+x
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        y = u+v+w+x
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)


def dobench_numpy2(size1, size2, dtype=np.float64):
    u = np.ones((size1, size2), dtype) * 0.1
    v = np.ones((size1, size2), dtype) * 0.2
    y = u+v
    t1 = time.time()
    for n in range(1000):
        y = u+v
    t2 = time.time()
    return (t2 - t1)

def dobench_numpy4(size1, size2, dtype=np.float64):
    u = np.array([[0.1] * size1] * size2, dtype)
    v = np.array([[0.2] * size1] * size2, dtype)
    w = np.array([[0.3] * size1] * size2, dtype)
    x = np.array([[0.4] * size1] * size2, dtype)
    y = u+v+w+x
    t1 = time.time()
    for n in range(1000):
        y = u+v+w+x
    t2 = time.time()
    return (t2 - t1)

     
dt = np.float64
print("dtype: %s; size = 2**n" % (np.dtype(dt).name))
print()
print("n\ty = x1 + x2\t\ty = x1 + x2 + x3 + x4\t\t(1000 times)")
for n in range(1,15):
    print("%d\t%s\t%s\t\tpyviennacl" % (n, 
                                        dobench2(2**n, 2**n, dt),
                                        dobench4(2**n, 2**n, dt) ))
    print("  \t%s\t%s\t\t_viennacl" % (dobench_vcl2(2**n, 2**n),
                                       dobench_vcl4(2**n, 2**n) ))
    print("  \t%s\t%s\t\tnumpy.array" % (dobench_numpy2(2**n, 2**n, dt),
                                         dobench_numpy4(2**n, 2**n, dt) ))
    print()
        
