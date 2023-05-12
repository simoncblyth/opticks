#!/usr/bin/env python
"""
U4SimulateTest_tt.py
========================

::

    u4t
    ./U4SimulateTest.sh tt
    ./U4SimulateTest.sh ntt

"""
import os, numpy as np, textwrap
from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.sysrap.sevt import SEvt


SCRIPT = "./U4SimulateTest.sh tt"
os.environ["SCRIPT"] = SCRIPT 
ENVOUT = os.environ.get("ENVOUT", None)
LABEL = os.environ.get("LABEL", "U4SimulateTest_tt.py" )


N = int(os.environ.get("N", "-1"))
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,-2,3,-3]
FLIP = int(os.environ.get("FLIP", "0")) == 1 
TIGHT = int(os.environ.get("TIGHT", "0")) == 1 

if MODE != 0:

    from opticks.ana.pvplt import * 
    WM = mp.pyplot.get_current_fig_manager()  
else:
    WM = None
pass

if __name__ == '__main__':

    print("AFOLD:%s" % os.environ.get("AFOLD", "-"))
    print("BFOLD:%s" % os.environ.get("BFOLD", "-"))
    print("N:%d " % N )

    if N == -1:
        a = SEvt.Load("$AFOLD",symbol="a")
        b = SEvt.Load("$BFOLD",symbol="b")
        syms = ['a','b']
        evts = [a,b]
    elif N == 0:
        a = SEvt.Load("$AFOLD",symbol="a")
        b = None
        syms = ['a']
        evts = [a,]
    elif N == 1:
        a = None
        b = SEvt.Load("$BFOLD",symbol="b")
        syms = ['b']
        evts = [b,]
    else:
        assert(0)
    pass

    if not a is None:print(repr(a))
    if not b is None:print(repr(b))


    EXPRS = r"""
    %(sym)s
    %(sym)s.symbol 
    w
    %(sym)s.q[w]
    %(sym)s.n[w]
    np.diff(%(sym)s.tt[w])
    np.c_[%(sym)s.t[w].view("datetime64[us]")] 
    """ 

    for sym in syms:

        n = eval("%(sym)s.n" % locals() ) 
        ww = np.where( n > 24)[0] 
   
        for w in ww:
            for e_ in textwrap.dedent(EXPRS).split("\n"):
                e = e_ % locals()
                print(e) 
                if len(e) == 0 or e[0] == "#": continue
                print(eval(e))
            pass
        pass
    pass













 
