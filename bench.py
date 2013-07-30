import pyviennacl as p
import time

def dobench2(size):
    u = p.Vector(size, 0.1)
    v = p.Vector(size, 0.2)
    p.Statement(u+v).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(10000):
        p.Statement(u+v).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

def dobench4(size):
    u = p.Vector(size, 0.1)
    v = p.Vector(size, 0.2)
    w = p.Vector(size, 0.3)
    x = p.Vector(size, 0.4)
    p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t1 = time.time()
    for n in range(10000):
        p.Statement(u+v+w+x).execute()
    p.backend_finish()
    t2 = time.time()
    return (t2 - t1)

for n in range(1,18):
    print("%d\t%s\t%s" % (n, dobench2(2**n), dobench4(2**n)))
