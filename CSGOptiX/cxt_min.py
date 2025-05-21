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



TODO: cherry pick from tests/CSGOptiXSimtraceTest.py and simplify
for minimal simtrace plotting

"""
import os, re, numpy as np
from collections import OrderedDict as odict
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt
from opticks.ana.p import cf    # load CSGFoundry geometry info

GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","2"))
NOGRID = "NOGRID" in os.environ
NOFRAME = "NOFRAME" in os.environ

if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
pass

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
    bn_u, bn_x, bn_n = np.unique(bn, return_index=True, return_counts=True)
    bn_o = np.argsort(bn_n)[::-1]

    _bn_tab = "np.c_[bn_u, bn_n, bn_x, cf.bdn[bn_u]][bn_o]"
    bn_tab = eval(_bn_tab)
    print("\n".join([_bn_tab,repr(bn_tab)]))


    """
    Form selection of simtrace intersects
    that have boundaries matching a regexp.
    Typically would have a few hundred boundary names
    in cf.bdn but potentially millions of simtrace intersects
    in the bn array.
    """

    bdict = odict()
    bdict['grey'] = '.*'
    bdict['red'] = 'Water/.*/Water$'
    bdict['blue'] = 'Water///Acrylic$'
    bdict['magenta'] = 'Water/Implicit_RINDEX_NoRINDEX_pOuterWaterInCD_pInnerReflector//Tyvek$'
    bdict['yellow'] = 'DeadWater/Implicit_RINDEX_NoRINDEX_pDeadWater_pTyvekFilm//Tyvek$'
    bdict['green'] = 'Water/Steel_surface//Steel'
    bdict['pink'] = 'Air/CDTyvekSurface//Tyvek$'
    bdict['cyan'] = 'Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterInCD_pCentralDetector/Water$'
    bdict['orange'] = 'Water/Steel_surface//Steel$'
    bdict['black'] = 'Air/CDTyvekSurface//Tyvek'

    d_qbn = cf.dict_find_boundary_indices_re_match(bdict)

    ## ax.scatter accepts color array, but its too slow with the large number of intersects needed
    wdict = {}
    for k,qbn in d_qbn.items():
        print("d_qbn[%s] %s \n" % (k, bdict[k]) ,d_qbn[k])
        print("\n".join(cf.bdn[d_qbn[k]]))
        print("\n")
        _w = np.isin( bn, d_qbn[k] )   # bool array indicating which elem of bn are in the qbn array
        w = np.where(_w)[0]       # indices of bn array with boundaries matching the regexp
        wdict[k] = w
    pass


    gpos = st[:,1].copy()
    gpos[...,3] = 1.
    lpos = np.dot( gpos, w2m )
    upos = gpos if GLOBAL else lpos

    H,V = 0,2

    if MODE == 2:
        pl = pvp.mpplt_plotter(label=label)
        fig, axs = pl
        ax = axs[0]

        for k, w in wdict.items():
            ax.scatter( upos[w,H], upos[w,V], s=0.1, color=k )
        pass

        if not NOGRID: ax.scatter( ugrid[:,H], ugrid[:,V], s=0.1, color="r" )

        #ax.set_ylim(-9000,2000)
        #ax.set_xlim(-22000, 22000)

        fig.show()
    elif MODE == 3:
        pl = pvp.pvplt_plotter(label)
        pvp.pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        if not NOFRAME: pvp.pvplt_frame(pl, sf, local=not GLOBAL )

        for k, w in wdict.items():
            pl.add_points(upos[w,:3], color=k )
        pass

        if not NOGRID: pl.add_points(ugrid[:,:3], color="r" )
        pl.show()
    else:
        pass
    pass
pass



