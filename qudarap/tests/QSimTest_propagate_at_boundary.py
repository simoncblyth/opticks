#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
#from opticks.ana.eprint import epr 

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")


def eprint( expr, lprefix="", rprefix="" ):
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, eval(expr) )
    print("%s : %s" % ( lhs, rhs )   )   

def epr(expr, prefix=""):
    ret = eval(expr)
    lhs = "%s%s" % (prefix, expr) 
    print("%-40s : %s " % (lhs, ret)) 
    return ret 



if __name__ == '__main__':
    t = Fold.Load(FOLD)

    p0 = t.p0 
    prd = t.prd 
    p = t.p 
    
    eprint("p0", rprefix="\n") 
    eprint("prd", rprefix="\n") 
    eprint("p", rprefix="\n") 

    flag = eprint("p[:,3,3].view(np.uint32)")
    eprint("np.unique(flag, return_counts=True)")





    
