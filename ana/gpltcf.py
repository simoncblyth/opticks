#!/usr/bin/env python

import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)

from opticks.analytic.gdml import GDML 
from opticks.ana.gplt import GPlot 


if __name__ == '__main__':

    plt.ion()
    args = GPlot.parse_args(__doc__)
    g0 = GDML.parse("$OPTICKS_PREFIX/tds0.gdml")
    g0.smry()
    g1 = GDML.parse("$OPTICKS_PREFIX/tds.gdml")
    g1.smry()

    lvx0 = "NNVTMCPPMT_PMT_20inch_log"
    lvx1 = "NNVTMCPPMT_log"

    lv0 = g0.find_one_volume(lvx0)
    log.info( "lv0 %r " % lv0 )

    lv1 = g1.find_one_volume(lvx1)
    log.info( "lv1 %r " % lv1 )


    lvs0 = g0.get_traversed_volumes( lv0, maxdepth=args.maxdepth )
    lvs1 = g1.get_traversed_volumes( lv1, maxdepth=args.maxdepth )


    fig, axs = GPlot.SubPlotsFig(plt, [lvs0,lvs1], args)
    fig.show()
    fig.savefig(args.pngpath("CF"))

    ax = axs[0,0]

    ax.set_xlim(0,100)
    ax.set_ylim(-250,-150)

    fig.show()
    fig.savefig(args.pngpath("CFZ"))




    #  -300,300  600 -> 200
    #  -400,200  600 -> 200  
    #
   
     




