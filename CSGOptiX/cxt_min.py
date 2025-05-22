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
from configparser import ConfigParser
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt
from opticks.ana.p import cf    # load CSGFoundry geometry info

GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","2"))
GSGRID = "GSGRID" in os.environ
FRAME = "FRAME" in os.environ

if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
pass



class CXT_Config(object):
    """
    Parses ini file of the below form::

        [key_boundary_regexp]
        red = (?P<Water_Virtuals>Water/.*/Water$)
        blue = Water///Acrylic$
        magenta = Water/Implicit_RINDEX_NoRINDEX_pOuterWaterInCD_pInnerReflector//Tyvek$
        yellow = DeadWater/Implicit_RINDEX_NoRINDEX_pDeadWater_pTyvekFilm//Tyvek$
        pink = Air/CDTyvekSurface//Tyvek$
        cyan = Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterInCD_pCentralDetector/Water$
        orange = Water/Steel_surface//Steel$
        grey =
        # empty value is special cased to mean all other boundary names

    Defaults path is $HOME/.opticks/GEOM/cxt_min.ini
    """

    @classmethod
    def Parse(cls, _path, _section ):
        cp = cls(_path, _section)
        return cp.bdict

    def __init__(self, _path, _section ):
        path = os.path.expandvars(_path)
        cfg = ConfigParser()
        cfg.read(path)
        sect = cfg[_section]
        bdict = OrderedDict(sect)

        self._path = _path
        self.path = path
        self._section = _section
        self.cfg = cfg
        self.bdict = bdict



if __name__ == '__main__':

    print("GLOBAL:%d MODE:%d" % (GLOBAL,MODE))

    e = SEvt.Load(symbol="e")   ## default load from FOLD envvar dir
    print(repr(e))
    label = e.f.base

    sf = e.f.sframe
    gs = e.f.genstep
    st = e.f.simtrace
    w2m = sf.w2m

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


    bn = st[:,2,3].view(np.int32)   ## simtrace intersect boundary indices
    btab = cf.boundary_table(bn)
    print(repr(btab))


    """
    Form selection of simtrace intersects
    that have boundaries matching a regexp.
    Typically would have a few hundred boundary names
    in cf.bdn but potentially millions of simtrace intersects
    in the bn array.
    """

    bdict = CXT_Config.Parse("$HOME/.opticks/GEOM/cxt_min.ini", "key_boundary_regexp")
    d_qbn, d_anno = cf.dict_find_boundary_indices_re_match(bdict)

    print("---------------------------------------------------\n")
    wdict = {}
    for k,qbn in d_qbn.items():

        qbn = d_qbn[k]   # boundary indices
        label = d_anno[k]
        bb = cf.bdn[qbn]

        print(" %10s %100s nbb:%4d %s " % (k, label, len(bb), str(qbn[:10])))
        #print("\n".join(cf.bdn[qbn]))
        #print("\n")
        _w = np.isin( bn, qbn )   # bool array indicating which elem of bn are in the qbn array
        w = np.where(_w)[0]       # indices of bn array with boundaries matching the regexp
        wdict[k] = w
    pass
    print("---------------------------------------------------\n")


    gpos = st[:,1].copy()
    gpos[...,3] = 1.
    lpos = np.dot( gpos, w2m )
    upos = gpos if GLOBAL else lpos

    X,Y,Z = 0,1,2
    H,V = X,Z

    if MODE == 2:
        pl = pvp.mpplt_plotter(label=label)
        fig, axs = pl
        ax = axs[0]

        for k, w in wdict.items():
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

        for k, w in wdict.items():
            label = d_anno[k]
            pl.add_points(upos[w,:3], color=k, label=label )
        pass

        if GSGRID: pl.add_points(ugrid[:,:3], color="r" )

        cp = pvp.pvplt_show(pl, incpoi=-5, legend=True )
    else:
        pass
    pass
pass



