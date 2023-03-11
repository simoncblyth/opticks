#!/usr/bin/env python
"""
U4SimulateTest_mtd.py
========================

::

    u4t
    ./U4SimulateTest.sh mtd

"""
import os, numpy as np
from opticks.ana.fold import Fold, AttrBase
from opticks.ana.p import * 

SURFACE_DETECT = 0x1 << 6 

LABEL = os.environ.get("LABEL", "U4SimulateTest_mtd.py" )

N = int(os.environ.get("VERSION", "-1"))
VERSION = N  
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,3]
PIDX = int(os.environ.get("PIDX", "123")) 


if MODE > 0:
    from opticks.ana.pvplt import * 
pass

from opticks.u4.tests.ModelTrigger_Debug import ModelTrigger_Debug       



if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    print("MODE:%d" % (MODE) )
    print("PIDX:%d" % (PIDX) )
    print("N:%d" % (N) )

    mtd = ModelTrigger_Debug(t, symbol="mtd", publish=False)  # publish:True crashing 
    print(mtd)


    SPECS = np.array(t.TRS_names.lines)
    st_ = t.aux[:,:,2,3].view(np.int32)
    st = SPECS[st_]

    u_st, n_st = np.unique(st, return_counts=True)
    expr = "np.c_[n_st,u_st][np.argsort(n_st)[::-1]]"
    print(expr)
    print(eval(expr))



    axes = 0, 2  # X,Z
    H,V = axes 
    label = LABEL 

    pos = t.photon[:,0,:3]
    q_ = t.seq[:,0]    #  t.seq shape eg (1000, 2, 2)  
    q = ht.seqhis(q_)    # history label eg b'TO BT BT SA ... lots of blankspace...'  
    n = np.sum( seqnib_(q_), axis=1 ) 


    expr = "q[PIDX]"
    print("\n%s ## " % expr)
    print(eval(expr))

   


    flagmask = t.photon[:,3,3].view(np.int32) 
    sd = flagmask & SURFACE_DETECT
    w_sd = np.where( sd )[0]


    x_midline = np.logical_and( pos[:,0] > -251, pos[:,0] < -249 )    
    z_midline = np.logical_and( pos[:,2] > -250, pos[:,2] <  250 )    
    xz_midline = np.logical_and( x_midline, z_midline )
    w_midline = np.where(xz_midline)[0]  


    ppos0 = pos
    ppos1 = pos[w_midline]







    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, ax = mpplt_plotter(label)

        ax.set_ylim(-250,250)
        ax.set_xlim(-500,500)

        if not ppos0 is None: ax.scatter( ppos0[:,H], ppos0[:,V], s=1 )  
        if not ppos1 is None: ax.scatter( ppos1[:,H], ppos1[:,V], s=1, c="r" )  


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
