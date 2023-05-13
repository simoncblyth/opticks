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


    label = "tt"

    fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
    ax = axs[0]

    e_ = "%(sym)s.s1 - %(sym)s.s0"
   

    bins = np.logspace( np.log10(0.1),np.log10(500.0), 25 ) 

    h = {} 
    for sym in syms: 
        e = e_ % locals() 
        h[sym] = np.histogram( eval(e) , bins=bins )   
        ax.plot( h[sym][1][:-1], h[sym][0], label=e  )
    pass
    ax.legend()
    fig.show()



    print("compare first and last point stamp range with beginPhoton endPhoton range")
    for sym in syms:
        nn_ = "np.arange( %(sym)s.n.min(), %(sym)s.n.max()+1, dtype=np.uint64 )" % locals() 
        print(nn_)
        nn = eval(nn_)
        for n in nn:  
            n_1 = int(n - 1)
            w_ = "np.where( %(sym)s.n == %(n)s )[0]" % locals()
            w = eval(w_)
            expr_ = "np.c_[%(sym)s.t[w,%(n_1)s] - %(sym)s.t[w,0], %(sym)s.ss[w], %(sym)s.ss[w] - %(sym)s.t[w,%(n_1)s] + %(sym)s.t[w,0] ]" % locals()
            expr = eval(expr_)
            print(w_)
            print(expr_)
            print(expr)
        pass
    pass
            









 
