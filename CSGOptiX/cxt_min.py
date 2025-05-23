#!/usr/bin/env python
"""
cxt_min.py
===========

1. loads simtrace SEvt from $FOLD
2. check validity with do asserts on the simtrace gensteps
3. prepare genstep ugrid positions and upos simtrace intersect positions
   depending on GLOBAL envvar

GLOBAL:1
   leave positions as is

GLOBAL:0
   apply the e.f.sframe.w2m (world2model) 4x4 transform to
   the global positions converting them into the model frame

4. plot the ugrid and upos points with in 2D with matplotlib MODE:2
   OR in 3D with pyvista MODE:3


::

    MODE=3 EYE=0,10000,0 ./cxt_min.sh pdb


Note issue that start with blank screen. Need to slighty
move the mouse wheel to get the render to appear.


TODO: cherry pick from tests/CSGOptiXSimtraceTest.py and simplify
for minimal simtrace plotting

"""
import os, re, logging, numpy as np
log = logging.getLogger(__name__)

from collections import OrderedDict
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt
from opticks.ana.p import cf    # load CSGFoundry geometry info
from opticks.CSG.CSGFoundry import BoundaryNameConfig

MODE = int(os.environ.get("MODE","2"))
NORMAL = int(os.environ.get("NORMAL","0"))
NORMAL_FILTER = int(os.environ.get("NORMAL_FILTER","10000"))

GLOBAL = 1 == int(os.environ.get("GLOBAL","0"))
GSGRID = 1 == int(os.environ.get("GSGRID","0"))
FRAME = 1 == int(os.environ.get("FRAME","0"))

if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
pass


class IdentityTable(object):
    def __init__(self, ii ):
        u, x, n = np.unique(ii, return_index=True, return_counts=True )
        o = np.argsort(n)[::-1]

        _tab = "np.c_[u,n,x][o]"
        tab = eval(_tab) 

        self._tab = _tab
        self.tab = tab

        self.u = u[o]
        self.x = x[o]
        self.n = n[o]

    def __repr__(self):
       lines =  []
       lines.append("[IdentityTable")
       lines.append(repr(self.tab))
       lines.append("]IdentityTable")
       return "\n".join(lines)


if __name__ == '__main__':

    print("GLOBAL:%d MODE:%d" % (GLOBAL,MODE))

    e = SEvt.Load("$FOLD", symbol="e")   ## default load from FOLD envvar dir

    print(repr(e))

    label = e.f.base
    sf = e.f.sframe
    w2m = sf.w2m

    gs = e.f.genstep
    st = e.f.simtrace

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

    st_x = st[:,1,0]
    st_y = st[:,1,1]
    st_z = st[:,1,2]
    st_id = st[:,3,3].view(np.int32) 

    presel = slice(None)                                        ## ALL

    #presel = st_x < -20000                                     ## x < -20m
    #presel = np.abs(st_x) < 1000                               ## |x|<1m 

    #presel = st_id == 0                                        ## non-instanced
    #presel = np.logical_and( st_id > 0, st_id < 30000 )        ## LPMT
    #presel = np.logical_and( st_id >= 30000, st_id < 50000 )   ## SPMT
    #presel = st_id >= 50000                                    ## pool PMTS
    


    ust = st[presel]   ## example ust shape (13812151, 4, 4)

    bn = ust[:,2,3].view(np.int32)   ## simtrace intersect boundary indices
    ii = ust[:,3,3].view(np.int32)

    iitab = IdentityTable(ii)
    print(repr(iitab))

    KEY = os.environ.get("KEY", None)
    KK = KEY.split(",") if not KEY is None else []


    cxtb, btab = cf.simtrace_boundary_analysis(bn, KEY)
    print(repr(cxtb))
    print(repr(btab))


    ## simtrace layout see sysrap/sevent.h
    gnrm = ust[:,0].copy()
    gpos = ust[:,1].copy()

    gnrm[...,3] = 0.   ## transform as vector
    gpos[...,3] = 1.   ## transform as position

    lnrm = np.dot( gnrm, w2m )
    lpos = np.dot( gpos, w2m )

    unrm = gnrm if GLOBAL else lnrm
    upos = gpos if GLOBAL else lpos



    X,Y,Z = 0,1,2
    H,V = X,Z


    if MODE == 2:
        pl = pvp.mpplt_plotter(label=label)
        fig, axs = pl
        ax = axs[0]

        for k, w in cxtb.wdict.items():
            if not KEY is None and not k in KK: continue
            ax.scatter( upos[w,H], upos[w,V], s=0.1, color=k )
        pass

        if GSGRID: ax.scatter( ugrid[:,H], ugrid[:,V], s=0.1, color="r" )

        #ax.set_ylim(-9000,2000)
        #ax.set_xlim(-22000, 22000)

        fig.show()
    elif MODE == 3:
        pl = pvp.pvplt_plotter(label)
        pvp.pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        if FRAME: pvp.pvplt_frame(pl, sf, local=not GLOBAL )

        for k, w in cxtb.wdict.items():
            if not KEY is None and not k in KK: continue
            label = "%10s %8d %s" % (k,len(w),cxtb.d_anno[k])
            pvp.pvplt_add_points(pl, upos[w,:3], color=k, label=label )

            if NORMAL > 0: pvp.pvplt_add_delta_lines(pl, upos[w,:3][::NORMAL_FILTER], 20*unrm[w,:3][::NORMAL_FILTER], color=k )
        pass

        if GSGRID: pl.add_points(ugrid[:,:3], color="r" )

        cp = pvp.pvplt_show(pl, incpoi=-5, legend=True )
    else:
        pass
    pass
pass



