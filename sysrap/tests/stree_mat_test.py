#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])

def Vacuum_kludge(ff, names=["mat","oldmat"]):
    for f in ff:
        print("Vacuum_kludge %s " % f.base)
        for m in names:
            q = getattr(f, m, None)
            if q is None: continue
            if np.all( q[16,0,:,1] == 1e9 ):
                print("%s : Vacuum 1e9 kludge reduce to 1e6 " % q )
                q[16,0,:,1] = 1e6 
            else:
                print("%s : Not doing Vacuum kludge" % q)
            pass
        pass
    pass



if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    Vacuum_kludge([t])


    t.mat[np.where( t.mat == 300. )] = 299.792458  # GROUPVEL default kludge 
    ab = np.abs( t.mat - t.oldmat ) 

    EXPR = """
    np.all( np.array( t.mat_names) == np.array( t.oldmat_names ))  
    t.mat.shape == t.oldmat.shape
    np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])
    np.array(t.mat_names)[np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])] 
    np.max(ab, axis=2).reshape(-1,8)   # max deviation across wavelength domain 
    #  RINDEX     ABSLENGTH  RAYLEIGH   REEMISSIONPROB   GROUPVEL 
    np.c_[np.arange(len(t.mat_names)),np.array(t.mat_names)] 
    """

    for expr in list(filter(None,textwrap.dedent(EXPR).split("\n"))):
        print(expr)
        if expr[0] == "#": continue
        print(eval(expr))
    pass

    wl = np.linspace(60.,820.,761)

    qwns="mat oldmat".split()

    MM = [4,11,14,17,18,19]

    for M in MM:
        MAT = t.mat_names[M]

        title = "GROUPVEL %d %s " % (M, MAT) 
        print(title)

        fig, ax = plt.subplots(1, figsize=SIZE/100.)
        fig.suptitle(title)

        for qwn in qwns:
            a = getattr(t, qwn, None)
            assert not a is None

            RINDEX = a[M,0,:,0]
            ABSLENGTH = a[M,0,:,1]
            RAYLEIGH = a[M,0,:,2]
            REEMISSIONPROB = a[M,0,:,3]
            GROUPVEL = a[M,1,:,0]   

            ax.plot( wl, GROUPVEL , label="GROUPVEL %s" % qwn )
        pass

        ax.legend()
        fig.show()
    pass

 



if 0:
    o = Fold.Load("/tmp/SBnd_test", symbol="o")
    print(repr(o))

    s = Fold.Load("/tmp/blyth/opticks/U4TreeCreateTest/stree", symbol="s")
    print(repr(s))

    Vacuum_kludge([t,o,s])


    exprs = """
    np.all( t.old_optical == o.optical )
    np.all( np.array( t.mat_names ) == np.array( o.mat_names ) ) 
    """

    for expr in list(filter(None,textwrap.dedent(exprs).split("\n"))):
        print(expr)
        print(eval(expr))
    pass

    assert len(t.mat) == len(o.mat)

    print("ij")
    for i in range(len(o.mat)): 
        for j in range(2):
            expr= "np.all( o.mat[%(i)d,%(j)d] == s.mat[%(i)d,%(j)d] )" % locals()
            print(" %s : %s " % (expr, eval(expr)))
        pass
    pass

    print("ijk")
    for i in range(len(o.mat)):
        tname = t.mat_names[i]
        oname = o.mat_names[i]
        assert( tname == oname )
        print( "\n i : %(i)d  %(tname)s " % locals() )
        for j in range(2):
            print( " j : %(j)d " % locals() )
            for k in range(4):
                expr= "len(np.where( np.abs( o.mat[%(i)d,%(j)d,:,%(k)d] - s.mat[%(i)d,%(j)d,:,%(k)d] ) > 1e-4)[0])" % locals()
                print(" %s : %s " % (expr, eval(expr)))
            pass
        pass
    pass
pass
 
