#!/usr/bin/env python
"""



"""
import os, logging, textwrap, numpy as np
from opticks.sysrap.sevt import SEvt

def eprint(expr, l, g ):
    print(expr)
    try:
       val = eval(expr,l,g)
    except AttributeError:
       val = "eprint:AttributeError"
    pass
    print(val)
pass
speed_ = lambda r,i:np.sqrt(np.sum( (r[:,i+1,0,:3]-r[:,i,0,:3])*(r[:,i+1,0,:3]-r[:,i,0,:3]), axis=1))/(r[:,i+1,0,3]-r[:,i,0,3])

if __name__ == '__main__':

    b = SEvt.Load("$FOLD", symbol="b")
    e = b 

    EXPR_ = r"""
    np.c_[np.unique(b.q, return_counts=True)] 
    """
    EXPR = list(filter(None,textwrap.dedent(EXPR_).split("\n")))
    for expr in EXPR:eprint(expr, locals(), globals() )
    
    select = "TO BT SA,TO BR BT SA"   ## select on photon history
    MSELECT = os.environ.get("SELECT", select )

    for SELECT in MSELECT.split(","): 

        context = "SELECT=\"%s\" ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh" % SELECT  
        print(context)
        SELECT_ELEM = SELECT.split()
        SELECT_POINT = len(SELECT_ELEM)

        sel_p = e.q_startswith(SELECT)
        sel_r = slice(0,SELECT_POINT)
        r = e.f.record[sel_p,sel_r]

        for i in range(SELECT_POINT-1):
            speed = speed_(r,i)
            speed_min = speed.min() if len(speed) > 0 else -1 
            speed_max = speed.max() if len(speed) > 0 else -1 
            fmt = "speed len/min/max for : %d -> %d : %s -> %s : %7d/%7.3f/%7.3f "
            print(fmt % (i,i+1,SELECT_ELEM[i],SELECT_ELEM[i+1], len(speed), speed_min, speed_max))
        pass
        print("\n".join(e.f.NPFold_meta.lines))   
    pass


