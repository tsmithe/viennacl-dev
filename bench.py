import numpy as np
import pyviennacl as p
import pyviennacl._viennacl as _v
import time

def dobench2(size, dtype=np.float64):
    u = p.Vector(size, 0.1, dtype=dtype)
    v = p.Vector(size, 0.2, dtype=dtype)
    p.Statement(u+v).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        p.Statement(u+v).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench4(size, dtype=np.float64):
    u = p.Vector(size, 0.1, dtype=dtype)
    v = p.Vector(size, 0.2, dtype=dtype)
    w = p.Vector(size, 0.3, dtype=dtype)
    x = p.Vector(size, 0.4, dtype=dtype)
    p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_vcl2(size):
    u = _v.vector_double(size, 0.1)
    v = _v.vector_double(size, 0.2)
    y = u+v
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        y = u+v
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_vcl4(size):
    u = _v.vector_double(size, 0.1)
    v = _v.vector_double(size, 0.2)
    w = _v.vector_double(size, 0.3)
    x = _v.vector_double(size, 0.4)
    y = u+v+w+x
    p.backend_finish()
    t1 = time.time()
    for n in range(1000):
        y = u+v+w+x
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_numpy2(size, dtype=np.float64):
    u = np.ones((size,), dtype) * 0.1
    v = np.ones((size,), dtype) * 0.2
    y = u+v
    t1 = time.time()
    for n in range(1000):
        y = u+v
    t2 = time.time()
    return (t2 - t1)

def dobench_numpy4(size, dtype=np.float64):
    u = np.ones((size,), dtype) * 0.1
    v = np.ones((size,), dtype) * 0.2
    w = np.ones((size,), dtype) * 0.3
    x = np.ones((size,), dtype) * 0.4
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
for n in range(1,19):
    print("%d\t%s\t%s\t\tpyviennacl" % (n, 
                                        dobench2(2**n, dt),
                                        dobench4(2**n, dt) ))
    print("  \t%s\t%s\t\t_viennacl" % (dobench_vcl2(2**n),
                                       dobench_vcl4(2**n) ))
    print("  \t%s\t%s\t\tnumpy.array" % (dobench_numpy2(2**n, dt),
                                         dobench_numpy4(2**n, dt) ))
    print()
        
