#!/usr/bin/env python
"""
G4CXTest_raindrop_simtrace.py
===============================

Simplify CSGOptiX/cxt_min.py


"""
import os, re, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.sysrap.sevt import SEvt
from opticks.ana.p import cf    # load CSGFoundry geometry info
from opticks.CSG.CSGFoundry import KeyNameConfig

GLOBAL = 1 == int(os.environ.get("GLOBAL","0"))
GSGRID = 1 == int(os.environ.get("GSGRID","0"))
FRAME = 1 == int(os.environ.get("FRAME","0"))
MODE = int(os.environ.get("MODE","2"))

X,Y,Z = 0,1,2
H,V = X,Z


if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
pass

if __name__ == '__main__':
    print("GLOBAL:%d MODE:%d" % (GLOBAL,MODE))

    t = SEvt.Load("$TFOLD", symbol="t")
    print(repr(t))

    label = t.f.base
    sf = t.f.sframe
    w2m = sf.w2m
    gs = t.f.genstep
    st = t.f.simtrace

    st_t = st[:,0,3]
    st_gp_bn = st[:,2,3].view(np.int32)
    st_gp = st_gp_bn >> 16  # globalPrimIdx
    st_bn = st_gp_bn & 0xffff   ## boundary


    gs_pos = gs[:,1]
    all_one = np.all( gs_pos[:,3] == 1. )
    all_zero = np.all( gs_pos[:,:3]  == 0 )
    assert all_one  # SHOULD ALWAYS BE 1.
    #assert all_zero # NOT ZERO WHEN USING CE_OFFSET=CE CE_SCALE=1
    gs_tra = gs[:,2:]
    assert gs_tra.shape[1:] == (4,4)

    ggrid = np.zeros( (len(gs), 4 ), dtype=np.float32 )
    for i in range(len(ggrid)): ggrid[i] = np.dot(gs_pos[i], gs_tra[i])
    ## np.matmul/np.tensordot/np.einsum can probably do this without the loop
    lgrid = np.dot( ggrid, w2m )
    ugrid = ggrid if GLOBAL else lgrid



    #presel = slice(None)
    #presel = st_t > 0.
    presel = st_gp > -1
    #presel = np.logical_and( st_gp > -1, st_t > 0. )



    ust = st[presel]

    gp_bn = ust[:,2,3].view(np.int32)    ## simtrace intersect boundary indices
    gp = gp_bn >> 16      ## globalPrimIdx
    bn = gp_bn & 0xffff   ## boundary


    u_gp, n_gp = np.unique(gp, return_counts=True )
    o_gp = np.argsort(n_gp)[::-1]
    gp_tab = np.c_[u_gp,n_gp][o_gp]
    print(repr(gp_tab))
    u_gp = gp_tab[:,0]
    n_gp = gp_tab[:,1]


    ## simtrace layout see sysrap/sevent.h
    gpos = ust[:,1].copy()
    gpos[...,3] = 1.   ## transform as position
    lpos = np.dot( gpos, w2m )
    upos = gpos if GLOBAL else lpos



    if MODE == 3:
        pl = pvp.pvplt_plotter(label)
        pvp.pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        if FRAME: pvp.pvplt_frame(pl, sf, local=not GLOBAL, pickable=False )


        colors = ["red", "green", "blue", "yellow", "magenta", "grey", "white" ]

        for i, g in enumerate(u_gp):
            w = np.where( gp == g )[0]
            color = colors[i % len(colors)]

            prn = cf.primname[g] if g > -1 and g < len(cf.primname) else "???"

            label = " gp:%d len(w):%d prn:%s col:%s " % ( g, len(w), prn, color )
            pvp.pvplt_add_points(pl, upos[w,:3], color=color, label=label )
        pass

        if GSGRID: pl.add_points(ugrid[:,:3], color="r", pickable=False )
        cp = pvp.pvplt_show(pl, incpoi=-5, legend=True, title=None )
    pass



