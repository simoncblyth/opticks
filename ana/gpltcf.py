#!/usr/bin/env python
"""
gpltcf.py
===========

::

   cp -r /tmp/fig/PolyconeNeck ~/simoncblyth.bitbucket.io/env/presentation/ana/
   l ~/simoncblyth.bitbucket.io/env/presentation/ana/PolyconeNeck/ 


"""
import matplotlib.pyplot as plt, numpy as np, logging
log = logging.getLogger(__name__)

from opticks.analytic.gdml import GDML 
from opticks.ana.gargs import GArgs 
from opticks.ana.gplt import GPlot 



if __name__ == '__main__':

    plt.ion()
    args = GArgs.parse(__doc__)

    #combiZoom = False
    combiZoom = True

    ilv = 1
    #ilv = 2

    lvx = args.lvname(ilv)
    if lvx.startswith("Hamamatsu"):
        def zoomlimits(ax):
            ax.set_xlim( 80,180)
            ax.set_ylim(-230,-130)
        pass
    elif lvx.startswith("NNVT"):
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




