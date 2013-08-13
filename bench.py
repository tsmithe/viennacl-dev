import numpy as np
import pyviennacl as p
import pyviennacl._viennacl as _v
import time

def dobench2(size1, size2, dtype=np.float64, ndim=1):
    if ndim == 2:
        y = p.Matrix(0.0, shape=(size1, size2), dtype=dtype)
        u = p.Matrix(size1, size2, 0.1, dtype=dtype)
        v = p.Matrix(size1, size2, 0.2, dtype=dtype)
    elif ndim == 1:
        y = p.Vector(size1, 0.0, dtype=dtype)
        u = p.Vector(size1, 0.1, dtype=dtype)
        v = p.Vector(size1, 0.2, dtype=dtype)        
    p.Statement(p.Assign(y, u*v)).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(100):
        p.Statement(p.Assign(y, u*v)).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench4(size1, size2, dtype=np.float64, ndim=1):
    if ndim == 2:
        y = p.Matrix(size1, size2, 0.0, dtype=dtype)
        u = p.Matrix(size1, size2, 0.1, dtype=dtype)
        v = p.Matrix(size1, size2, 0.2, dtype=dtype)
        w = p.Matrix(size1, size2, 0.3, dtype=dtype)
        x = p.Matrix(size1, size2, 0.4, dtype=dtype)
    elif ndim == 1:
        y = p.Vector(size1, 0.0, dtype=dtype)
        u = p.Vector(size1, 0.1, dtype=dtype)
        v = p.Vector(size1, 0.2, dtype=dtype)
        w = p.Vector(size1, 0.3, dtype=dtype)
        x = p.Vector(size1, 0.4, dtype=dtype)
    p.Statement(p.Assign(y, u*v*w*x)).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(100):
        p.Statement(p.Assign(y, u*v*w*x)).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_vcl2(size1, size2, ndim=1):
    if ndim == 2:
        u = _v.matrix_double(size1, size2, 0.1)
        v = _v.matrix_double(size1, size2, 0.2)
    elif ndim == 1:
        u = _v.vector_double(size1, 0.1)
        v = _v.vector_double(size1, 0.2)
    y = u+v
    p.backend_finish()
    t1 = time.time()
    for n in range(100):
        y = u+v
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench_vcl4(size1, size2, ndim=1):
    if ndim == 2:
        u = _v.matrix_double(size1, size2, 0.1)
        v = _v.matrix_double(size1, size2, 0.2)
        w = _v.matrix_double(size1, size2, 0.3)
        x = _v.matrix_double(size1, size2, 0.4)
    elif ndim == 1:
        u = _v.vector_double(size1, 0.1)
        v = _v.vector_double(size1, 0.2)
        w = _v.vector_double(size1, 0.3)
        x = _v.vector_double(size1, 0.4)
    y = u+v+w+x
    p.backend_finish()
    t1 = time.time()
    for n in range(100):
        y = u+v+w+x
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)


def dobench_numpy2(size1, size2, dtype=np.float64, ndim=1):
    if ndim == 2:
        u = np.matrix(np.ones((size1, size2), dtype) * 0.1)
        v = np.matrix(np.ones((size1, size2), dtype) * 0.2)
    elif ndim == 1:
        u = np.matrix(np.ones((size1, ), dtype) * 0.1)
        v = np.matrix(np.ones((size1, ), dtype) * 0.2)
    y = u*v
    t1 = time.time()
    for n in range(100):
        y = u*v
    t2 = time.time()
    return (t2 - t1)

def dobench_numpy4(size1, size2, dtype=np.float64, ndim=1):
    if ndim == 2:
        u = np.matrix([[0.1] * size1] * size2, dtype)
        v = np.matrix([[0.2] * size1] * size2, dtype)
        w = np.matrix([[0.3] * size1] * size2, dtype)
        x = np.matrix([[0.4] * size1] * size2, dtype)
    elif ndim == 1:
        u = np.matrix([0.1] * size1, dtype)
        v = np.matrix([0.2] * size1, dtype)
        w = np.matrix([0.3] * size1, dtype)
        x = np.matrix([0.4] * size1, dtype)
    y = u*v*w*x
    t1 = time.time()
    for n in range(100):
        y = u*v*w*x
    t2 = time.time()
    return (t2 - t1)

     
dt = np.float32
ndim = 2
print("dtype: %s; ndim = %d; size = 2**n" % (np.dtype(dt).name, ndim))
print()
print("n\ty = x1 + x2\t\ty = x1 + x2 + x3 + x4\t\t(100 times)")
for n in range(1,30):
    print("%d\t%s\t%s\t\tpyviennacl" % (n, 
                                        dobench2(2**n, 2**n, dt, ndim),0))#,
                                        #dobench4(2**n, 2**n, dt, ndim) ))
#    print("  \t%s\t%s\t\t_viennacl" % (dobench_vcl2(2**n, 2**n, ndim),
#                                       dobench_vcl4(2**n, 2**n, ndim) ))
    print("  \t%s\t%s\t\tnumpy.array" %(dobench_numpy2(2**n,2**n,dt,ndim),0))#,
                                         #dobench_numpy4(2**n,2**n,dt,ndim)))
    print()
        
