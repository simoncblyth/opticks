#!/usr/bin/env python
"""
U4SimulateTest_mtd.py
========================

::

    u4t
    ./U4SimulateTest.sh mtd

"""
import os, textwrap, numpy as np
from opticks.ana.fold import Fold, AttrBase
from opticks.ana.p import * 

SURFACE_DETECT = 0x1 << 6 
SURFACE_ABSORB = 0x1 << 7

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

    SPECS = np.array(t.U4R_names.lines) 
    st_ = t.aux[:,:,2,3].view(np.int32)
    st = SPECS[st_]

    u_st, n_st = np.unique(st, return_counts=True)
    expr = "np.c_[n_st,u_st][np.argsort(n_st)[::-1]]"
    print(expr)
    print(eval(expr))



    axes = 0, 2  # X,Z
    H,V = axes 
    label = LABEL 

    end = t.photon[:,0,:3]
    q_ = t.seq[:,0]    #  t.seq shape eg (1000, 2, 2)  
    q = ht.seqhis(q_)    # history label eg b'TO BT BT SA ... lots of blankspace...'  
    n = np.sum( seqnib_(q_), axis=1 ) 


    exprs = "q[PIDX] t.record[PIDX,:n[PIDX],0] mtd.pv[mtd.index==PIDX]"    
    for expr in exprs.split():
        print("\n%s ## " % expr)
        print(eval(expr))
    pass

    exprs = "np.unique(mtd.whereAmI[mtd.trig==1],return_counts=True)"   
    for expr in exprs.split():
        print("\n%s ## " % expr)
        print(eval(expr))
    pass

    
    mtd_trig = mtd.trig == 1 
    mtd_pyrex  = np.logical_and(mtd.trig == 1, mtd.whereAmI_ == 1 )
    mtd_vacuum = np.logical_and(mtd.trig == 1, mtd.whereAmI_ == 2 )
    mtd_outside = np.logical_and(mtd.trig == 1, mtd.EInside1 == 0 )

    mtd_upper = mtd.pos[:,2] > 1e-4   
    mtd_vacuum_upper = np.logical_and(mtd_vacuum, mtd_upper )

    idxs = np.unique(mtd.index[mtd_vacuum_upper])   # photon indices with vac on upper 


    flagmask = t.photon[:,3,3].view(np.int32) 
    sd = flagmask & SURFACE_DETECT != 0 
    sa = flagmask & SURFACE_ABSORB != 0 



    ## HMM: this is specific to midline of the left hand PMT in twopmt_layout 
    x_midline = np.logical_and( end[:,0] > -251, end[:,0] < -249 )    
    z_midline = np.logical_and( end[:,2] > -250, end[:,2] <  250 )    
    xz_midline = np.logical_and( x_midline, z_midline )
    w_midline = np.where(xz_midline)[0]  

    #ppos0_ = "None"
    #ppos0_ = "end"
    #ppos0_ = "end[sd] # photon SD endpoints around the upper hemi"
    #ppos0_ = "end[sa] # photon SA endpoints around the upper hemi and elsewhere"
    #ppos0_ = "end[w_midline]  # photons ending on midline " 
    #ppos0_ = "mtd.pos[mtd_outside] # just around upper hemi "
    #ppos0_ = "mtd.pos[mtd_pyrex]  # just around upper hemi "
    #ppos0_ = "mtd.pos[mtd_vacuum] # mostly on midline, sprinkle of obliques around upper hemi "
    ppos0_ = "mtd.next_pos[mtd_trig] ## just around upper hemi"

    ppos1_ = "None" 
    #ppos1_ = "mtd.pos[mtd_vacuum_upper]"


    ppos0  = eval(ppos0_)
    ppos1  = eval(ppos1_) 
    label = "b:%s\nr:%s" % ( ppos0_, ppos1_ )


    exprs = textwrap.dedent("""
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==2),mtd.EInside1==0))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==2),mtd.EInside1==1))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==2),mtd.EInside1==2))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==1),mtd.EInside1==0))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==1),mtd.EInside1==1))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==1),mtd.EInside1==2))
    """)

    for expr in exprs.split():
        print(expr)
        print(eval(expr))
    pass


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
