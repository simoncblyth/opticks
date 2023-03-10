#!/usr/bin/env python
"""
U4SimulateTest_mtd.py
========================

::

    u4t
    ./U4SimulateTest.sh mtd

"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

SURFACE_DETECT = 0x1 << 6 

LABEL = os.environ.get("LABEL", "U4SimulateTest_mtd.py" )
VERSION =  int(os.environ.get("VERSION", "-1"))
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,3]
PIDX = int(os.environ.get("PIDX", "123")) 


if MODE > 0:
    from opticks.ana.pvplt import * 
pass

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    print( " MODE : %d " % (MODE) )
    print( " PIDX : %d " % (PIDX) )

    mtd = t.ModelTrigger_Debug  
    PV = np.array(t.ModelTrigger_Debug_meta.PV) 
    MLV = np.array(t.ModelTrigger_Debug_meta.MLV) 
    WAI = np.array( ["OutOfRegion", "kInGlass   ", "kInVacuum  ", "kUnset     "] )

    mt_pos  = mtd[:,0,:3]
    mt_time = mtd[:,0,3]

    mt_dir  = mtd[:,1,:3]
    mt_energy = mtd[:,1,3]

    mt_dist1 = mtd[:,2,0]
    mt_dist2 = mtd[:,2,1]
    mt_mlv_   = mtd[:,2,2].view(np.uint64) 
    mt_etrig = mtd[:,2,3].view("|S8") 

    mt_index     = mtd[:,3,0].view(np.uint64)  # photon index for each candidate ModelTrigger
    mt_pv_       = mtd[:,3,1].view(np.uint64) 
    mt_whereAmI_ = mtd[:,3,2].view(np.uint64) 
    mt_trig      = mtd[:,3,3].view(np.uint64) 

    mt_next_pos  = mtd[:,4,:3]
    mt_next_mct  = mtd[:,4,3]
    mt_next_norm = mtd[:,5,:3]


    mt_mlv = MLV[mt_mlv_]
    mt_pv  = PV[mt_pv_]
    mt_whereAmI = WAI[mt_whereAmI_]

    mt_dist2[mt_dist2 == 9e99] = np.inf  ## avoid obnoxious 9e99 kInfinity
    mt_dist1[mt_dist1 == 9e99] = np.inf

    tr = np.array([[0.000,0.000,-1.000,0.000],[0.000,1.000,0.000,0.000],[1.000,0.000,0.000,0.000],[-250.000,0.000,0.000,1.000]],dtype=np.float64)

    lpos = np.ones( (len(mt_pos),4) )
    lpos[:,:3] = mt_pos

    ldir = np.zeros( (len(mt_dir),4) )
    ldir[:,:3] = mt_dir

    mt_gpos = np.dot( lpos, tr )  
    mt_gdir = np.dot( ldir, tr )

    lnext_pos = np.ones( (len(mt_next_pos),4) )
    lnext_pos[:,:3] = mt_next_pos

    lnext_norm = np.zeros( (len(mt_next_norm),4) )
    lnext_norm[:,:3] = mt_next_norm

    mt_gnext_pos  = np.dot( lnext_pos , tr )  
    mt_gnext_norm = np.dot( lnext_norm, tr )



    expr = "np.c_[mt_index, mt_whereAmI, mt_trig, mt_etrig, mt_pv, mt_mlv][mt_index == PIDX]"
    print("\n%s ## ModelTrigger_Debug mlv and pv for PIDX " % expr)
    print(eval(expr))

    print("## kInVacuum : ACTUALLY pv is inner1_phys ")
    print("## kInGlass  : ACTUALLY pv NOT inner1_phys ")
    print("## kUnset    : ACTUALLY pv is inner2_phys causing early exit ")

    expr = " np.c_[mt_index, mt_pos[:,2],mt_time, mt_gpos[:,:3], mt_gdir[:,:3], mt_dist1, mt_dist2][mt_index == PIDX] "
    print("\n%s ## ModelTrigger_Debug for PIDX " % expr)
    print(eval(expr))

    if 0:
        expr = "np.c_[lpos,mt_gpos]"
        print("\n%s ## " % expr)
        print(eval(expr))

        expr = "np.c_[ldir,mt_gdir]"
        print("\n%s ## " % expr)
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
