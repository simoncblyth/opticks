#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.eprint import eprint, epr

if __name__ == '__main__':
    FOLD = os.environ["FOLD"]
    t = Fold.Load(FOLD) 

    g = globals()
    l = locals()


    epr("FOLD",          g, l )
    p0  = epr("t.p0",    g, l,  tail="\n\n", rprefix="\n"  ) 
    prd = epr("t.prd",   g, l,  tail="\n\n", rprefix="\n"  ) 
    s   = epr("t.s",     g, l,  tail="\n\n", rprefix="\n"  ) 
    p   = epr("t.p",     g, l,  tail="\n\n", rprefix="\n"  ) 

    epr("FOLD", g, l )

    assert not p is None
    flag       = epr("flag=p[:,3,3].view(np.uint32)",             g, l )
    uflag      = epr("uflag=np.unique(flag, return_counts=True)", g, l )
    flat       = epr("flat=p[:,0,3]",                             g, l, tail="HMM: sphoton.time not flat" )
    TransCoeff = epr("TransCoeff=p[:,1,3]",                       g, l, tail="HMM: sphoton.weight not TransCoeff" ) 



    
