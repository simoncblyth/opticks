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
from opticks.CSG.CSGFoundry import KeyNameConfig

MODE = int(os.environ.get("MODE","2"))
NORMAL = int(os.environ.get("NORMAL","0"))
NORMAL_FILTER = int(os.environ.get("NORMAL_FILTER","10000"))

GLOBAL = 1 == int(os.environ.get("GLOBAL","0"))
GSGRID = 1 == int(os.environ.get("GSGRID","0"))
FRAME = 1 == int(os.environ.get("FRAME","0"))

if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
pass



class UniqueTable(object):
    def __init__(self, symbol, ii, names=None ):
        u, x, c = np.unique(ii, return_index=True, return_counts=True )
        o = np.argsort(c)[::-1]
        nn = names[u] if not names is None else None

        _tab = "np.c_[u,c,x,nn][o]" if not nn is None else "np.c_[u,c,x][o]"
        tab = eval(_tab)

        self.symbol = symbol
        self._tab = _tab
        self.tab = tab

        self.u = u[o]
        self.x = x[o]
        self.c = c[o]
        self.n = nn[o] if not nn is None else None


    def __repr__(self):
       lines =  []
       lines.append("[%s" % self.symbol)
       lines.append(repr(self.tab))
       lines.append("]%s" % self.symbol)
       return "\n".join(lines)




if __name__ == '__main__':

    print("GLOBAL:%d MODE:%d" % (GLOBAL,MODE))

    prn_config = KeyNameConfig.Parse("$HOME/.opticks/GEOM/cxt_min.ini", "key_prim_regexp")


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
    st_gp = st[:,2,3].view(np.int32) >> 16
    st_bn = st[:,2,3].view(np.int32) & 0xffff
    st_id = st[:,3,3].view(np.int32)

    presel = slice(None)                                        ## ALL

    #presel = st_x < -20000                                     ## x < -20m
    #presel = np.abs(st_x) < 1000                               ## |x|<1m

    PRESEL = os.environ.get("PRESEL", "")
    if PRESEL.startswith("PRIM:"):
        spec = PRESEL[len("PRIM:"):]
        gps = np.array([], dtype=np.int64)
        for elem in spec.split(","):
            if str.isdigit(elem):
                gps = np.concatenate(gps, int(elem))
            else:
                egp = np.unique(np.where( cf.primname == elem ))
                gps = np.concatenate( (gps, egp) )
            pass
        pass
        gps = np.unique(gps)
        presel = np.isin(st_gp, gps )
        pass
    elif PRESEL == "LPMT":
        presel = np.logical_and( st_id > 0, st_id < 30000 )
    elif PRESEL == "SPMT":
        presel = np.logical_and( st_id >= 30000, st_id < 50000 )
    elif PRESEL == "PPMT":
        presel = st_id >= 50000   ## pool PMTS
    elif PRESEL == "GLOBAL":
        presel = st_id == 0   ## non-instanced
    else:
        presel = slice(None)
    pass

    #presel = st_gp == 10
    #presel = st_gp == 2919
    #presel = st_gp == 2919



    ust = st[presel]   ## example ust shape (13812151, 4, 4)

    gp_bn = ust[:,2,3].view(np.int32)    ## simtrace intersect boundary indices
    gp = gp_bn >> 16      ## globalPrimIdx
    bn = gp_bn & 0xffff   ## boundary

    ii = ust[:,3,3].view(np.int32)   ## instanceIndex

    idtab = UniqueTable("idtab", ii, None)
    print(repr(idtab))

    #gptab = UniqueTable("gptab", gp, cf.primname)
    #print(repr(gptab))

    ## would be good to see the globalPrimIdx that correspond to a boundary

    KEY = os.environ.get("KEY", None)
    KK = KEY.split(",") if not KEY is None else []

    cxtb, btab = cf.simtrace_boundary_analysis(bn, KEY)
    print(repr(cxtb))
    print(repr(btab))

    cxtp, ptab = cf.simtrace_prim_analysis(gp, KEY)
    print(repr(cxtp))
    print(repr(ptab))

    PRIMTAB = "PRIMTAB" in os.environ
    cxtable = cxtp if PRIMTAB else cxtb


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


    class SimtracePlot(object):
        """
        defined inline as uses::

             KEY
             upos
             unrm
             NORMAL_FILTER

        """
        def __init__(self, pl, cxtable):
            self.cxtable = cxtable
            pl.enable_point_picking(callback=self, use_picker=True, show_message=False)
            pcloud = {}
            pcinfo = np.zeros( (len(cxtable.wdict),3), dtype=np.int64 )
            i = 0
            keys = np.array(list(cxtable.wdict.keys()))
            for k, w in cxtable.wdict.items():
                if not KEY is None and not k in KK: continue
                label = self.get_label(k)
                pcloud[k] = pvp.pvplt_add_points(pl, upos[w,:3], color=k, label=label )
                n_verts = pcloud[k].n_verts if not pcloud[k] is None else 0
                pcinfo[i,0] = n_verts
                if NORMAL > 0: pvp.pvplt_add_delta_lines(pl, upos[w,:3][::NORMAL_FILTER], 20*unrm[w,:3][::NORMAL_FILTER], color=k, pickable=False )
                i += 1
            pass
            pcinfo[:,1] = np.cumsum(pcinfo[:,0])
            pcinfo[:,2] = pcinfo[:,0] + pcinfo[:,1] - 1
            self.pcloud = pcloud
            self.pcinfo = pcinfo
            self.keys = keys

        def get_label(self, k):
            cxtable = self.cxtable
            w = cxtable.wdict[k]
            label = "%10s %8d %s" % (k,len(w),cxtable.d_anno[k])
            return label

        def __call__(self, picked_point, picker):
            """
            """
            self.picked_point = picked_point
            self.picker = picker

            pcloud = self.pcloud
            pcinfo = self.pcinfo

            dataSet = picker.GetDataSet()
            pointId = picker.GetPointId()

            idx= list(pcloud.values()).index(dataSet)
            k = list(pcloud.keys())[idx]
            label = self.get_label(k)

            print("SimtracePlot.__call__ picked_point %s pointId %d idx %d k %s label %s " % (str(picked_point), pointId, idx, k, label) )
        pass
    pass

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
        if FRAME: pvp.pvplt_frame(pl, sf, local=not GLOBAL, pickable=False )

        spl = SimtracePlot(pl, cxtable)

        if GSGRID: pl.add_points(ugrid[:,:3], color="r", pickable=False )

        cp = pvp.pvplt_show(pl, incpoi=-5, legend=True, title=None )
    else:
        pass
    pass
pass


