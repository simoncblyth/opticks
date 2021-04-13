#!/usr/bin/env python
"""
gpltcf.py
===========

This plots solids parsed from two GDML files providing 
a visualization of geometry changes.

See also:

gpmt.py 
    plotting solids from a single GDML file


::

   cp -r /tmp/fig/PolyconeNeck ~/simoncblyth.bitbucket.io/env/presentation/ana/
   l ~/simoncblyth.bitbucket.io/env/presentation/ana/PolyconeNeck/ 


::

   jcv HamamatsuR12860PMTManager
   jcv Hamamatsu_R12860_PMTSolid


"""
import matplotlib.pyplot as plt, numpy as np, logging
log = logging.getLogger(__name__)

from opticks.analytic.gdml import GDML 
from opticks.ana.gargs import GArgs 
from opticks.ana.gplt import GPlot 

from j.PMTEfficiencyCheck_ import PMTEfficiencyCheck_ 

if __name__ == '__main__':

    plt.ion()

    pec = PMTEfficiencyCheck_()
    #pec = None

    args = GArgs.parse(__doc__)

    combiZoom = False
    #combiZoom = True

    #ilv = 1    # NNVT
    ilv = 2     # Hama

    if ilv == 1:
        ipec = 1     # i:1 NNVTMCPPMT_PMT_20inch_body_phys 
    elif ilv == 2:
        ipec = 0     # i:0 HamamatsuR12860_PMT_20inch_body_phys
    else:
        ipec = None
    pass


    closeup = False

    lvx = args.lvname(ilv)
    if lvx.startswith("Hamamatsu") and closeup:
        def zoomlimits(ax):
            ax.set_xlim( 80,180)
            ax.set_ylim(-230,-130)
        pass
    elif lvx.startswith("NNVT") and closeup:
        def zoomlimits(ax):
            ax.set_xlim(  40,140)
            ax.set_ylim(-250,-150)
        pass
    else:
        def zoomlimits(ax):
            pass
        pass
    pass 

    args.figdir = "/tmp/fig/PolyconeNeck"

    figpfx = "gpltcf"
    figpfx += "_%s" % lvx
    if combiZoom: figpfx += "_combiZoom" 

    args.figpfx = figpfx
    args.shorten_title_ = lambda s:s.replace("PMT_20inch_","")  


    g0 = GDML.parse(args.gdmlpath(0))
    g0.smry()
    g1 = GDML.parse(args.gdmlpath(1))
    g1.smry()

    lv0 = g0.find_one_volume(lvx)
    log.info( "lv0 %r " % lvx )

    lv1 = g1.find_one_volume(lvx)
    log.info( "lv1 %r " % lvx )

    lvs0 = g0.get_traversed_volumes( lv0, maxdepth=args.maxdepth )
    lvs1 = g1.get_traversed_volumes( lv1, maxdepth=args.maxdepth )

    fig, axs = GPlot.SubPlotsFig(plt, [lvs0,lvs1], args, combiZoom=combiZoom, zoomlimits=zoomlimits)

    if not pec is None:
        pec.rz_plot(axs, ipec) 
    pass

    fig.show()

    cfpath = args.figpath("_SubPlotsFig")
    log.info("save to cfpath : %s " % cfpath)
    fig.savefig(cfpath)


    if combiZoom == False:
        ax = axs[0,0] if len(axs.shape) == 2 else axs[0]
        zoomlimits(ax)

        fig.show()

        cfzpath = args.figpath("_SubPlotsFig_Zoom")
        log.info("save to cfzpath : %s " % cfzpath)
        fig.savefig(cfzpath)




