#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

def eprint( expr, lprefix="", rprefix="", tail="" ):
    ret = eval(expr)
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, ret )
    print("%-50s : %s%s" % ( lhs, rhs, tail )   )   
    return ret 

def epr(arg, **kwa):
    p = arg.find("=")  
    if p > -1:
        var_eq = arg[:p+1]
        expr = arg[p+1:]
        label = var_eq
    else:
        label, expr = "", arg 
    pass
    return eprint(expr, lprefix=label,  **kwa)


if __name__ == '__main__':
    FOLD = os.environ["FOLD"]
    epr("FOLD")
    t = Fold.Load(FOLD) 

    if hasattr(t,'p0'):
        p0 = t.p0 
        epr("p0", tail="\n\n", rprefix="\n"  ) 
    else:
        p0 = None
    pass

    if hasattr(t,'prd'):
        prd = t.prd 
        epr("prd", tail="\n\n", rprefix="\n" ) 
    else:
        prd = None
    pass

    if hasattr(t,'s'):
        s = t.s 
        epr("s", tail="\n\n", rprefix="\n" ) 
    else:
        s = None
    pass


    if hasattr(t,'p'):
        p = t.p 
        epr("p", tail="\n\n", rprefix="\n" ) 
    else:
        p = None
    pass


    assert not p is None

   
