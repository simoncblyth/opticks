#!/usr/bin/env python
"""
U4SimulateTest_pr.py
========================

::

    u4t
    ./U4SimulateTest.sh pr
    ./U4SimulateTest.sh npr

"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.u4.tests.U4SimulateTest import U4SimulateTest


SCRIPT = "./U4SimulateTest.sh pr"
os.environ["SCRIPT"] = SCRIPT 
ENVOUT = os.environ.get("ENVOUT", None)


N = int(os.environ.get("N", "-1"))
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,-2,3,-3]
FLIP = int(os.environ.get("FLIP", "0")) == 1 
TIGHT = int(os.environ.get("TIGHT", "0")) == 1 

if MODE != 0:
    from opticks.ana.pvplt import * 
pass

if __name__ == '__main__':

    if N == -1:
        a = U4SimulateTest.Load("$AFOLD",symbol="a")
        b = U4SimulateTest.Load("$BFOLD",symbol="b")
        syms = ['a','b']
    elif N == 0:
        a = U4SimulateTest.Load("$AFOLD",symbol="a")
        b = None
        syms = ['a']
    elif N == 1:
        a = None
        b = U4SimulateTest.Load("$BFOLD",symbol="b")
        syms = ['b']
    else:
        assert(0)
    pass

    if not a is None:print(repr(a))
    if not b is None:print(repr(b))
 
    print( "MODE:%d" % (MODE) )

    if FLIP == False:
        H,V = 0, 2        # customary X, Z 
    else:
        H,V = 2, 0        # flipped   Z, X 
    pass
    
    scale = 0.8 if TIGHT else 1.0

    LIM = { 
            0:np.array([-500,500])*scale,
            1:np.array([-500,500])*scale,
            2:np.array([-250,250])*scale
          }


    #ppos0_ = "None"
    #ppos0_ = "pos #   "
    ppos0_ = "t.f.record[:,0,0,:3] # 0-position   "

    #ppos1_ = "None"
    ppos1_ = "t.f.record[:,1,0,:3] # 1-position   "

    #ppos2_ = "None"
    ppos2_ = "t.f.record[t.n>2,2,0,:3] # 2-position   "

    #ppos3_ = "None"
    ppos3_ = "t.f.photon[:,0,:3]"

    tlabel_ = "t.TITLE"  
    tid_ = "t.ID"  


    elem = []
    if not ppos0_ is "None": elem.append("b:%s" % ppos0_)
    if not ppos1_ is "None": elem.append("r:%s" % ppos1_)
    if not ppos2_ is "None": elem.append("g:%s" % ppos2_)
    if not ppos3_ is "None": elem.append("c:%s" % ppos3_)

    ppos0 = {}
    ppos1 = {}
    ppos2 = {}
    ppos3 = {}
    tlabel = {}
    tid = {}

    for i,sym in enumerate(syms):
        t = eval(sym)
        ppos0[sym] = eval(ppos0_)
        ppos1[sym] = eval(ppos1_) 
        ppos2[sym] = eval(ppos2_) 
        ppos3[sym] = eval(ppos3_) 
        tlabel[sym] = eval(tlabel_) 
        tid[sym] = eval(tid_) 
    pass

    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        for sym in syms:
            if TIGHT: 
                label = tlabel[sym]
            else:
                label = "\n".join([tlabel[sym]] + elem)
            pass
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label)
            ax = axs[0]
            ax.set_ylim(*LIM[V])
            ax.set_xlim(*LIM[H])
            if TIGHT:ax.axis('off')

            if not ppos0[sym] is None: ax.scatter( ppos0[sym][:,H], ppos0[sym][:,V], s=1, c="b" )  
            if not ppos1[sym] is None: ax.scatter( ppos1[sym][:,H], ppos1[sym][:,V], s=1, c="r" )  
            if not ppos2[sym] is None: ax.scatter( ppos2[sym][:,H], ppos2[sym][:,V], s=1, c="g" )  
            if not ppos3[sym] is None: ax.scatter( ppos3[sym][:,H], ppos3[sym][:,V], s=1, c="c" )  
            pass
            if TIGHT:fig.tight_layout()

            if not ENVOUT is None:
                envout = "\n".join([
                               "export ENVOUT_PATH=%s" % ENVOUT,
                               "export ENVOUT_SYM=%s" % sym,
                               "export ENVOUT_TID=%s" % tid[sym],
                               "export ENVOUT_VERSION=%s" % tid[sym],
                               ""
                               ]) 
                open(ENVOUT, "w").write(envout)
                print(envout)
            pass

            fig.show()
        pass
    elif MODE == -2:
        ## TRYING TO SHOW MORE THAN ONE PLOT GIVES SUBPLOTS TOO SMALL 
        ## ITS MORE USEFUL TO POP UP TWO WINDOWS AS DONE ABOVE 
        fig, axs = mpplt_plotter(nrows=1, ncols=2, label=label)
        for i,ax in enumerate(axs):
            sym = syms[i]

            ax.set_ylim(*LIM[V])
            ax.set_xlim(*LIM[H])
            if TIGHT:ax.axis('off')

            if not ppos0[sym] is None: ax.scatter( ppos0[sym][:,H], ppos0[sym][:,V], s=1, c="b" )  
            if not ppos1[sym] is None: ax.scatter( ppos1[sym][:,H], ppos1[sym][:,V], s=1, c="r" )  
            if not ppos2[sym] is None: ax.scatter( ppos2[sym][:,H], ppos2[sym][:,V], s=1, c="g" )  
            if not ppos3[sym] is None: ax.scatter( ppos3[sym][:,H], ppos3[sym][:,V], s=1, c="c" )  
        pass
        if TIGHT:fig.tight_layout()
        fig.show()
    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points(ppos )
        pl.show()
    pass
pass
