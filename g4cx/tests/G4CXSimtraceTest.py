#!/usr/bin/env python
"""
G4CXSimtraceTest.py
=====================

TODO: further reduce duplication with cx/tests/CSGOptiXSimtraceTest.py  

Used from gxt.sh, eg::

   SIMPLE=1 ./gxt.sh ana
   ./gxt.sh ana

   MASK=non ./gxt.sh ana
       when the genstep grid is too small the view with default MASK=pos
       will be very zoomed in, use MASK=non to not mask the positions plotted
       by the genstep grid points 

   MASK=t ./gxt.sh ana
       curious MASK=t is unexpected making half the hama_body_log disappear ? 


"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import *       # including cf loaded from CFBASE

from opticks.ana.feature import SimtraceFeatures
from opticks.ana.framegensteps import FrameGensteps
from opticks.ana.simtrace_positions import SimtracePositions
from opticks.ana.simtrace_plot import SimtracePlot, pv, mp
from opticks.ana.pvplt import *


SIMPLE = "SIMPLE" in os.environ
MASK = os.environ.get("MASK", "pos")
FEAT = os.environ.get("FEAT", "pid" )

def simple(pl, gs, pos):
    """ 
    :param pl: pvplt_plotter instance
    :param gs: FrameGensteps instance
    :param pos: SimtracePositions instance
    """
    pvplt_simple(pl, gs.centers_local[:,:3], "gs.centers_local[:,:3]" )   
    pvplt_simple(pl, pos.gpos[:,:3], "pos.gpos[:,:3]" )
    pvplt_simple(pl, pos.lpos[:,:3], "pos.lpos[:,:3]" )


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    x = None
    print(t)

    local = True 

    gs = FrameGensteps(t.genstep, t.sframe, local=local, symbol="gs" )  ## get gs positions in target frame
    print(gs)

    t_pos = SimtracePositions(t.simtrace, gs, t.sframe, local=local, mask=MASK, symbol="t_pos" )
    print(t_pos)

    if SIMPLE:
        pl = pvplt_plotter()
        simple(pl, gs, t_pos)
        pl.show()
        raise Exception("SIMPLE done")
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


