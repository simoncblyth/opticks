#!/usr/bin/env python
"""
cxt_min.py
===========

1. loads simtrace SEvt from $MFOLD
2. checks validity with asserts on the simtrace gensteps
3. prepare genstep ugrid positions and upos simtrace intersect positions
   depending on GLOBAL envvar
4. plot the ugrid and upos points


MODE=2
   use matplotlib 2D plotting, little used now : as lacks flexibility and performance

MODE=3
   use pyvista 3D plotting.
   Have previously observed pyvista issue that sometimes start with a blank screen.
   It is necessary to slighty move the mouse wheel to get the render to appear.

GLOBAL=1
   leave positions as is resulting in global frame plotting

   With global frame the plane of the genstep rays will often not correspond
   to the plane of the view causing the geometry to appear squeezed until
   rotate around with pyvista (MODE:3) manually.
   For the same reason it is difficult to target using EYE
   when using GLOBAL frame.

   TODO: provide option to handle EYE,LOOK,UP inputs as local frame
   even when using GLOBAL frame ?

GLOBAL=0
   apply the e.f.sframe.w2m (world2model) 4x4 transform to
   the global positions converting them into the model frame


GSGRID=1
   plot the grid of genstep positions

PRIMTAB=1
   configure identity coloring and key labelling of intersects to use globalPrimIdx
   (this succeeded to color both Opticks and Geant4 intersects)

PRIMTAB=0 (default)
   configure identity coloring and key labelling of intersects to use boundary index
   (note that when plotting U4Simtrace.h Geant4 simtrace intersects the boundary is
   currently always zero)


KEY=red,white,powderblue
   select intersects included in the plot according to their key color

KEY=~red,white,powderblue
   select intersects excluded from the plot according to their key color


MODE=3 EYE=0,10000,0
   EYE controls 3D viewpoint in pyvista plotting using mm length units,
   NB unlike cxr_min.sh there is currently no target extent scaling so
   the EYE,LOOK,UP values need to be chosen according to the size
   of the target geometry


TODO: cherry pick from tests/CSGOptiXSimtraceTest.py and simplify
for minimal simtrace plotting




This python script is also used from ipc InputPhotonCheck, eg::

    ipc env_info ## get into env
    MODE=3 PRIMTAB=1 GSGRID=1 EYE=0,1000,0  ipc cxt             ## viewpoint 1m offset in Y from center of target frame
    MODE=3 PRIMTAB=1 GSGRID=1 EYE=0,1000,0 GLOBAL=1 ipc cxt


    MODE=3 PRIMTAB=1 GSGRID=1 EYE=0,1000,0 GLOBAL=0 KEY=red,white,powderblue ipc cxt
    MODE=3 PRIMTAB=1 GSGRID=1 EYE=0,1000,0 GLOBAL=0 KEY=~red,white,powderblue ipc cxt


"""
import os, sys, re, logging, textwrap, numpy as np
log = logging.getLogger(__name__)

from collections import OrderedDict
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt
from opticks.ana.p import cf    # load CSGFoundry geometry info
from opticks.CSG.CSGFoundry import KeyNameConfig

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None
pass


MODE = int(os.environ.get("MODE","2"))
NORMAL = int(os.environ.get("NORMAL","0"))
NORMAL_FILTER = int(os.environ.get("NORMAL_FILTER","10000")) # modulo scaledown for normal presentation
PRIMTAB = int(os.environ.get("PRIMTAB","0"))
assert PRIMTAB in [0,1]


A = int(os.environ.get("A","0"))
B = int(os.environ.get("B","0"))


GLOBAL = 1 == int(os.environ.get("GLOBAL","0"))
GSGRID = 1 == int(os.environ.get("GSGRID","0"))
FRAME = 1 == int(os.environ.get("FRAME","0"))
POINT = np.fromstring(os.environ["POINT"],sep=",").reshape(-1,3) if "POINT" in os.environ else None
POINTSIZE = float(os.environ.get("POINTSIZE",1.))
APOINTSIZE = float(os.environ.get("APOINTSIZE",10.))
BPOINTSIZE = float(os.environ.get("BPOINTSIZE",10.))


if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
pass



class UniqueTable(object):
    def __init__(self, symbol, ii, names=None ):
        u, x, c = np.unique(ii, return_index=True, return_counts=True )
        o = np.argsort(c)[::-1] ## descending count order
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


    if not "MFOLD" in os.environ:
       print("FATAL cxt_min.py simtrace plotting now needs MFOLD envvar, not the former FOLD : as that is too common")
    pass
    e = SEvt.Load("$MFOLD", symbol="e")
    a = SEvt.Load("$AFOLD", symbol="a") if A == 1 else None
    b = SEvt.Load("$BFOLD", symbol="b") if B == 1 else None


    print(repr(e))
    if e is None:
        fmt = "ABORT script as failed to SEvt.Load from MFOLD[%s] GEOM is [%s]"
        print(fmt % (os.environ.get("MFOLD", None), os.environ.get("GEOM", None)))
        sys.exit(0)
    pass

    label = e.f.base
    sf = e.f.sframe
    w2m = sf.w2m
    m2w = sf.m2w

    gs = e.f.genstep
    st = e.f.simtrace

    gs_pos = gs[:,1]
    all_one = np.all( gs_pos[:,3] == 1. )    ## W
    all_zero = np.all( gs_pos[:,:3]  == 0 )  ## XYZ often all zero
    ## the gs_pos is regarded as a local position in the
    ## frame defined by the transform in the last 4 quads of the simtrace genstep

    assert all_one  # SHOULD ALWAYS BE 1. [HOMOGENOUS POSITION COORDINATE W :  4th ELEMENT]
    #assert all_zero # NOT ZERO WHEN USING CE_OFFSET=CE CE_SCALE=1
    gs_tra = gs[:,2:]
    assert gs_tra.shape[1:] == (4,4)  ## skip first number of gs dimensions


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

    EXPR = list(map(str.strip,textwrap.dedent(r"""
    st.shape
    ust.shape
    np.c_[np.unique(gp,return_counts=True)]
    np.c_[np.unique(bn,return_counts=True)]
    np.c_[np.unique(ii,return_counts=True)]
    """).split("\n")))

    for expr in EXPR:
        print(expr)
        if expr == "" or expr[0] == "#": continue
        print(repr(eval(expr)))
    pass


    idtab = UniqueTable("idtab", ii, None)
    print(repr(idtab))

    gptab = UniqueTable("gptab", gp, cf.primname)
    print(repr(gptab))

    ## would be good to see the globalPrimIdx that correspond to a boundary

    KEY = os.environ.get("KEY", None)
    KEY_invert = False
    if not KEY is None:
        if KEY.startswith("~"):
            KEY = KEY[1:]
            KEY_invert = True
        pass
        KK = KEY.split(",")
    else:
        KK = None
    pass

    ## analyse frequencies of occurence of the label integers
    cxtb, btab = cf.simtrace_boundary_analysis(bn, KEY)
    print(repr(cxtb))
    print(repr(btab))

    cxtp, ptab = cf.simtrace_prim_analysis(gp, KEY)
    print(repr(cxtp))
    print(repr(ptab))

    cxtable = cxtp if PRIMTAB==1 else cxtb


    ## for simtrace layout see sysrap/sevent.h
    gnrm = ust[:,0].copy()
    gpos = ust[:,1].copy()

    gnrm[...,3] = 0.   ## surface normal transforms as vector
    gpos[...,3] = 1.   ## intersect position transforms as position

    ## transform from global to local frame
    lnrm = np.dot( gnrm, w2m )
    lpos = np.dot( gpos, w2m )

    elu_m2w = m2w if GLOBAL else np.eye(4)   ## try to make EYE,LOOK,UP stay local even in GLOBAL
    unrm = gnrm if GLOBAL else lnrm
    upos = gpos if GLOBAL else lpos

    ASLICE = None
    AIDX = int(os.environ.get("AIDX","0"))
    if not a is None:
        if "AFOLD_RECORD_SLICE" in os.environ:
            ASLICE = os.environ["AFOLD_RECORD_SLICE"]
            ase = a.q_startswith(ASLICE)
            g_apos = a.f.record[ase][:,:,0].copy()
        else:
            ase = np.where( a.f.record[:,:,3,3].view(np.int32) > 0 )   ## flagmask selection to skip ufilled step point records
            g_apos = a.f.record[ase][:,0].copy()
            ## HMM: how to avoid loosing separation between photon records
        pass

        g_apos[...,3] = 1.  # transform as position
        l_apos = np.dot( g_apos, w2m )
        u_apos = g_apos if GLOBAL else l_apos
    else:
        g_apos = None
        l_apos = None
        u_apos = None
    pass

    if not b is None:
        if "BFOLD_RECORD_SLICE" in os.environ:
            bse = b.q_startswith(os.environ["BFOLD_RECORD_SLICE"])
            g_bpos = b.f.record[bse][:,:,0].copy()
        else:
            bse = np.where( b.f.record[:,:,3,3].view(np.int32) > 0 )   ## flagmask selection to skip ufilled step point records
            g_bpos = b.f.record[bse][:,0].copy()
        pass
        g_bpos[...,3] = 1.  # transform as position
        l_bpos = np.dot( g_bpos, w2m )
        u_bpos = g_bpos if GLOBAL else l_bpos
    else:
        g_bpos = None
        l_bpos = None
        u_bpos = None
    pass



    X,Y,Z = 0,1,2
    H,V = X,Z



    class SimtracePlot(object):
        """
        defined inline as uses::

             KK
             upos
             unrm
             NORMAL_FILTER

        """
        def __init__(self, pl, cxtable, hide=False):
            """
            :param pl: pyvista plotter
            :param cxtable: analysis of simtrace integer such as boundary or globalPrimIdx
            """
            self.cxtable = cxtable
            pl.enable_point_picking(callback=self, use_picker=True, show_message=False)
            pcloud = {}
            pts = {}
            nrm = {}
            pcinfo = np.zeros( (len(cxtable.wdict),3), dtype=np.int64 )
            i = 0
            keys = np.array(list(cxtable.wdict.keys()))
            #print("SimtracePlot keys:%s" % str(keys))
            for k, w in cxtable.wdict.items():
                if not KK is None:
                    skip0 = KEY_invert is False and not k in KK
                    skip1 = KEY_invert is True and k in KK
                    if skip0: continue
                    if skip1: continue
                pass
                label = self.get_label(k)
                pts[k] = upos[w,:3]
                nrm[k] = unrm[w,:3]
                if not hide:
                    pcloud[k] = pvp.pvplt_add_points(pl, pts[k], color=k, label=label )
                else:
                    pcloud[k] = None
                pass
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
            self.pts = pts
            self.nrm = nrm

        def get_label(self, k):
            cxtable = self.cxtable
            w = cxtable.wdict[k]
            label = "%15s %8d %s" % (k,len(w),cxtable.d_anno[k])
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

        pvp.pvplt_viewpoint(pl, verbose=True, m2w=elu_m2w)   # sensitive to EYE, LOOK, UP envvars


        if FRAME: pvp.pvplt_frame(pl, sf, local=not GLOBAL, pickable=False )


        HIDE="HIDE" in os.environ
        spl = SimtracePlot(pl, cxtable, hide=HIDE)


        if "OVERLAP" in os.environ and len(spl.pcloud) == 2 and not KDTree is None:
            overlap_tol = float(os.environ["OVERLAP"]) # eg 1 0.1 0.01 0.001 or 1e-3
            assert overlap_tol > 0
            close_tol = overlap_tol*10
            print("OVERLAP check overlap_tol : %s close_tol %s " % (overlap_tol, close_tol  ))
            k0, k1 = list(spl.pts.keys())
            pt0 = spl.pts[k0]    # eg (547840, 3)
            pt1 = spl.pts[k1]    # eg (259548, 3)
            nr0 = spl.nrm[k0]    # eg (547840, 3)
            nr1 = spl.nrm[k1]    # eg (259548, 3)

            print("-OVERLAP-form KDTree kt0")
            kt0 = KDTree(pt0)
            print("-OVERLAP-form KDTree kt1")
            kt1 = KDTree(pt1)

            # query nearest neighbors between two clouds
            print("-OVERLAP-kt0.query")
            dist_1to0, idx_1to0 = kt0.query(pt1, k=1) # pt1: (259548, 3) dist_1to0: (259548,)  idx_1to0:(259548,)
            print("-OVERLAP-kt1.query")
            dist_0to1, idx_0to1 = kt1.query(pt0, k=1) #                  dist_0to1: (547840,)

            close_pt0 = pt0[dist_0to1 < close_tol]
            close_pt1 = pt1[dist_1to0 < close_tol]
            close_pt = np.unique(np.vstack([close_pt0, close_pt1]), axis=0)

            all_sd0 = np.sum( (pt0 - pt1[idx_0to1])*nr1[idx_0to1], axis=1 )
            all_sd1 = np.sum( (pt1 - pt0[idx_1to0])*nr0[idx_1to0], axis=1 )

            overlap_pt0 = pt0[all_sd0 < -overlap_tol]
            overlap_pt1 = pt1[all_sd1 < -overlap_tol]

            overlap_pt = np.unique(np.vstack([overlap_pt0, overlap_pt1]), axis=0)
            np.save("/tmp/overlap_pt.npy", overlap_pt)

            close_label = "close_pt  %s " % (len(close_pt))
            overlap_label = "overlap_pt  %s " % (len(overlap_pt))

            #pvp.pvplt_add_points(pl, close_pt,   color="yellow", label=close_label,   copy_mesh=True, point_size=5 ) # HUH setting pointsize here effects all points
            pvp.pvplt_add_points(pl, overlap_pt, color="yellow", label=overlap_label, copy_mesh=True, point_size=5 ) # HUH setting pointsize here effects all points
        else:
            close_pd = None
        pass



        if GSGRID: pl.add_points(ugrid[:,:3], color="r", pickable=False )

        if not u_apos is None: pvp.pvplt_add_points_adaptive(pl, u_apos, color="red",  label="u_apos" ) # point_size=APOINTSIZE )
        if not u_bpos is None: pvp.pvplt_add_points_adaptive(pl, u_bpos, color="blue", label="u_bpos" ) # point_size=BPOINTSIZE )

        if not u_apos is None and u_apos.ndim == 3 and not ASLICE is None:
            num_step_point = len(ASLICE.split())
            pvp.pvplt_add_contiguous_line_segments( pl, u_apos[AIDX,:num_step_point,:3], point_size=4 )
        pass

        def callback_click_position(xy):
            print("callback_click_position xy : %s " % str(xy))
        pass
        pl.track_click_position(callback_click_position)
        if not POINT is None: pl.add_points(POINT, color="r", label="POINT") # point_size=POINTSIZE, render_points_as_spheres=True)

        #pl.reset_camera() # avoids blank start screen, but resets view to include all points - not what want

        cp = pvp.pvplt_show(pl, incpoi=-5, legend=True, title=None )
    else:
        pass
    pass
pass


