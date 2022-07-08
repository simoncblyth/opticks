#!/usr/bin/env python
"""
tests/CSGOptiXSimtraceTest.py
==============================

* see notes/issues/simtrace-shakedown.rst


See also:

csg/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh


This allows interactive visualization of workstation 
generated intersect data fphoton.npy on remote machines such as 
user laptops that support pyvista. 


FEAT envvar controlling intersect coloring and legend titles
--------------------------------------------------------------

pid
    uses cf.primIdx_meshname_dict()
bnd
    uses cf.sim.bndnamedict
ins
    uses cf.insnamedict

    instance identity : less different feature colors normally 
    but interesting to see what is in which instance and what is in ins0 the global instance, 
    the legend names are for example : ins37684 ins42990 ins0 ins43029


ISEL envvar selects simtrace geometry intersects by their features, according to frequency order
-------------------------------------------------------------------------------------------------------

FEAT=ins ISEL=0
    only show the instance with the most intersects
FEAT=ins ISEL=0,1
    only show the two instances with the most and 2nd most intersects
FEAT=bnd ISEL=0,1
    ditto for boundaries

FEAT=pid ISEL=0,1



pyvista GUI keys
----------------------

* https://docs.pyvista.org/api/plotting/plotting.html

* to zoom out/in : slide two fingers up/down on trackpad. 
* to pan : hold down shift and one finger tap-lock, then move finger around  


Too many items in the legend
-----------------------------

When not using MASK=pos the legend may be filled with feature item lines 
that are not visible in the frame 


FramePhotons vs Photons
---------------------------

Using frame photons is a trick to effectively see results 
from many more photons that have to pay the costs for transfers etc.. 
Frame photons lodge photons onto a frame of pixels limiting 
the maximumm number of photons to handle. 

ISEL allows plotting of a selection of feature values only, picked by descending frequency index
-------------------------------------------------------------------------------------------------

::

    cx ; ./cxs_Hama.sh  grab
    cx ; ./cxs_Hama.sh  ana 

Old instructions, not recently exercised::

    cx ; ./cxs_grab.sh   ## NO LONGER USED ?

    ISEL=0,1         ./cxs.sh    # ISEL=0,1 picks the 2 most frequent feature values (eg boundaries when FEAT=bnd)
    ISEL=0,1,2,3,4   ./cxs.sh 

    ISEL=Hama        ./cxs.sh    # select boundaries via strings in the bndnames, assuming FEAT=bnd
    ISEL=NNVT        ./cxs.sh 
    ISEL=Pyrex       ./cxs.sh 
    ISEL=Pyrex,Water ./cxs.sh 


"""
import os, sys, logging, numpy as np
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)
log = logging.getLogger(__name__)

from opticks.ana.eget import efloatlist_, elookce_, elook_epsilon_, eint_

SIZE = np.array([1280, 720])   ## SIZE*2 [2560, 1440]
XCOMPARE_SIMPLE = "XCOMPARE_SIMPLE" in os.environ
XCOMPARE = "XCOMPARE" in os.environ
GUI = not "NOGUI" in os.environ
MP =  not "NOMP" in os.environ 
PV =  not "NOPV" in os.environ 
PVGRID = not "NOPVGRID" in os.environ
LEGEND =  not "NOLEGEND" in os.environ # when not MASK=pos legend often too many lines, so can switch it off 
SIMPLE = "SIMPLE" in os.environ
MASK = os.environ.get("MASK", "pos")
FEAT = os.environ.get("FEAT", "pid" )  
ALLOWED_MASK = ("pos", "t", "non" )
assert MASK in ALLOWED_MASK, "MASK %s is not in ALLOWED_MASK list %s " % (MASK, str(ALLOWED_MASK))
GSPLOT = eint_("GSPLOT", "0")
PIDX = eint_("PIDX", "0")


from opticks.CSG.CSGFoundry import CSGFoundry 
from opticks.ana.p import *       # including cf loaded from CFBASE
from opticks.ana.fold import Fold
from opticks.ana.feature import SimtraceFeatures
from opticks.ana.simtrace_positions import SimtracePositions
from opticks.ana.simtrace_plot import SimtracePlot
from opticks.ana.framegensteps import FrameGensteps
from opticks.ana.npmeta import NPMeta
from opticks.sysrap.sframe import sframe , X, Y, Z
from opticks.ana.pvplt import * 

import matplotlib
if GUI == False:
    log.info("set pdf backend as GUI False")
    matplotlib.use("agg")
pass

if MP:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        mp = None
    pass
else:
    mp = None
pass

if PV:
    try:
        import pyvista as pv
        themes = ["default", "dark", "paraview", "document" ]
        pv.set_plot_theme(themes[1])
    except ImportError:
        pv = None
    pass
else:
    pv = None
pass

if GUI == False:
    log.info("disabling pv as GUI False")
    pv = None
pass



def pvplt_simple(xyz, label):
    """
    :param xyz: (n,3) shaped array of positions
    :param label: to place on plot 

    KEEP THIS SIMPLE : FOR DEBUGGING WHEN LESS BELLS AND WHISTLES IS AN ADVANTAGE
    """
    pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
    pl.add_text( "pvplt_simple %s " % label, position="upper_left")
    pl.add_points( xyz, color="white" )        
    pl.show_grid()
    cp = pl.show() if GUI else None
    return cp



def xcompare_simple( pos, x_gpos, x_lpos, local=True ):
    """
    :param pos: SimtracePositions instance
    :param x_gpos: global photon step positions
    :param x_lpos: local photon step positions
    """
    pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
    if local == False:
        pl.add_points( pos.gpos[:,:3], color="white" )        
        pvplt_add_contiguous_line_segments(pl, x_gpos[:,:3])
    else:
        pl.add_points( pos.lpos[:,:3], color="white" )        
        pvplt_add_contiguous_line_segments(pl, x_lpos[:,:3])
    pass
    pl.show_grid()

    outpath = "/tmp/xcompare_simple.png" 
    log.info("outpath %s " % outpath)
    #pl.screenshot(outpath)  segments

    cp = pl.show(screenshot=outpath) if GUI else None
    return pl 


def simple(pl, pos):
    """
    :param pos: SimtracePositions instance
    """
    pvplt_simple(pl, pos.gpos[:,:3], "pos.gpos[:,:3]" )
    pvplt_simple(pl, pos.lpos[:,:3], "pos.lpos[:,:3]" )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    t = Fold.Load("$CFBASE/CSGOptiXSimtraceTest", symbol="t"); 
    x = Fold.Load("$CFBASE/CSGOptiXSimTest", symbol="x")

    if not x is None:
        x_nib = seqnib_(x.seq[:,0])  # valid steppoint records from seqhis count_nibbles
        x_gpos_ = x.record[PIDX,:x_nib[PIDX],0,:3]  # global frame photon step record positions of single PIDX photon
        x_gpos  = np.ones( (len(x_gpos_), 4 ), dtype=np.float32 )
        x_gpos[:,:3] = x_gpos_
        x_lpos = np.dot( x_gpos, t.sframe.w2m ) 
    pass

    SimtracePositions.Check(t.simtrace)

    local = True 

    gs = FrameGensteps(t.genstep, t.sframe, local=local, symbol="gs" )  ## get gs positions in target frame

    t_pos = SimtracePositions(t.simtrace, gs, t.sframe, local=local, mask=MASK, symbol="t_pos" )

    if SIMPLE:
       pl = pvplt_plotter()
       simple(pl, t_pos)
       pl.show()
       raise Exception("SIMPLE done")
    pass
    if XCOMPARE_SIMPLE and not x is None:
       pl = xcompare_simple( t_pos, x_gpos, x_lpos, local=True )
       #raise Exception("XCOMPARE done")
    pass

    pf = SimtraceFeatures(t_pos, cf, featname=FEAT, symbol="pf" ) 

    pl = SimtracePlot.MakePVPlotter()

    plt = SimtracePlot(pl, pf.feat, gs, t.sframe, t_pos, outdir=os.path.join(t.base, "figs") )

    if not x is None:       
        plt.x_lpos = x_lpos   
    pass

    if not mp is None:
        plt.positions_mpplt()
        ax = plt.ax
    pass

    if not pv is None:
        plt.positions_pvplt()
    pass
pass
