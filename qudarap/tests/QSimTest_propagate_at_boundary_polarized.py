#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.eprint import epr 

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")


def epr(expr, prefix=""):
    ret = eval(expr)
    lhs = "%s%s" % (prefix, expr) 
    print("%-40s : %s " % (lhs, ret)) 
    return ret 



if __name__ == '__main__':
    t = Fold.Load(FOLD)

    p = t.p 

    print(p) 

    print("\n\n")

    flag = epr("p[:,3,3].view(np.uint32)", "flag=")

    epr("np.unique(flag, return_counts=True)")

    TransCoeff = epr("p[:,1,3]", "TransCoeff=")

    flat = epr("p[:,0,3]", "flat=")





    
