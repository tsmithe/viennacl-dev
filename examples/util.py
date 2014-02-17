import pyviennacl as p

def read_mtx(fname):
    """
    Read a MatrixMarket file. Assume coordinate format, and double precision.
    
    Very crude!
    """
    fd = open(fname)
    lines = map(lambda x: x.strip().split(" "), fd.readlines())
    ln = 0
    for line in lines:
        ln += 1
        if line[0][0] == "%":
            continue
        else:
            break
    n = int(lines[ln][0])
    m = int(lines[ln][1])
    nnz = int(lines[ln][2])
    mat = p.CompressedMatrix(n, m, nnz)
    mat_type = p.np_result_type(mat).type
    def assign(l):
        i, j, v = int(l[0]), int(l[1]), mat_type(l[2])
        mat[i, j] = v
        print(i, j, v, mat[i,j])
    map(assign, lines[ln+1:])
    print(mat)
    #print(mat.nonzeros)
    #print(mat.nnz)
    return mat
