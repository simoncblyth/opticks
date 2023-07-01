#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold

def Vacuum_kludge(ff):
    for f in ff:
        print("Vacuum_kludge %s " % f.base)
        if np.all( f.mat[16,0,:,1] == 1e9 ):
            print("Vacuum 1e9 kludge reduce to 1e6 : because it causes obnoxious presentation")
            f.mat[16,0,:,1] = 1e6 
        else:
            print("Not doing Vacuum kludge")
        pass
    pass



if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

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
 
