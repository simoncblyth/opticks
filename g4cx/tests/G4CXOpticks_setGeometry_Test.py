#!/bin/bash -l 

import numpy as np, textwrap
from opticks.CSG.CSGFoundry import CSGFoundry

if __name__ == '__main__':
    cf = CSGFoundry.Load("$FOLD") 
    print(repr(cf))

    boundary = cf.node[:,1,2].view(np.int32) 

    u_bnd, n_bnd = np.unique(boundary, return_counts=True )
    order = np.argsort(n_bnd)[::-1]

    bn = cf.sim.stree.standard.bnd_names
    l_bnd = bn[u_bnd]

    abn = cf.sim.stree.standard.bnd_names
    idx = np.arange(len(abn))

    compare_stree = False
    if compare_stree:
        ## compare bnd_names between the GGeo stree and the standard one
        ## TODO: get rid of the GGeo stree ?
        bbn = cf.sim.extra.GGeo.bnd_names
        wdif = np.where( abn != bbn )[0]   
        # same counts and isur suppression off no-RINDEX is the only difference when OSUR is disabled
        # large number of osur are different when OSUR is enabled
        assert len(abn) == len(bbn)
        EXPR = filter(None,textwrap.dedent(r"""
        np.c_[u_bnd,n_bnd,l_bnd][order]
        np.c_[idx,cf.sim.extra.GGeo.bnd_names,cf.sim.stree.standard.bnd_names]
        wdif
        np.c_[cf.sim.extra.GGeo.bnd_names,cf.sim.stree.standard.bnd_names][wdif]
        """).split())
    else:
        EXPR = filter(None,textwrap.dedent(r"""
        np.c_[u_bnd,n_bnd,l_bnd][order]
        np.c_[idx,cf.sim.stree.standard.bnd_names]
        """).split())
    pass

    for expr in EXPR:
        print("\n")
        print(expr)
        print(repr(eval(expr)))
    pass
    pass     
    print(cf.descSolids())   
pass 

    
