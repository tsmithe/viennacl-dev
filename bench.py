import numpy as np
import pyviennacl as p
import time

def dobench2(size, dtype=np.float64):
    u = p.Vector(size, 0.1, dtype=dtype)
    v = p.Vector(size, 0.2, dtype=dtype)
    p.Statement(u+v).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(10000):
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
    for n in range(10000):
        p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

for n in range(1,18):
    print("%d\t%s\t%s" % (n, 
                          dobench2(2**n, np.float32),
                          dobench4(2**n, np.float32)))
