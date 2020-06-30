#!/usr/bin/env python

import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)

from opticks.analytic.gdml import GDML 
from opticks.ana.gplt import GPlot 


if 0:
    lvx = specs_(r"""
    PMT_3inch_log
    NNVTMCPPMTlMaskVirtual
    HamamatsuR12860lMaskVirtual
    mask_PMT_20inch_vetolMaskVirtual
    """
    )
    ilv = 3

    labels = specs_(r"""
    tds_ngt
    tds_ngt_pcnk
    """
    )


if __name__ == '__main__':

    plt.ion()
    args = GPlot.parse_args(__doc__)

    ##  label/lvx : specify "label.gdml" and logical volume name prefix 
    #spec = ["tds0/NNVTMCPPMT_PMT_20inch_log","tds/NNVTMCPPMT_log" ]

    spec = ["tds_ngt/NNVTMCPPMT_PMT_20inch_log","tds_ngt_pcnk/NNVTMCPPMT_PMT_20inch_log" ]
    #spec = ["tds_ngt/HamamatsuR12860_PMT_20inch_log","tds_ngt_pcnk/HamamatsuR12860_PMT_20inch_log" ]

    label = map(lambda s:s.split("/")[0], spec )
    lvx = map(lambda s:s.split("/")[1], spec )



    g0 = GDML.parse("$OPTICKS_PREFIX/%s.gdml" % label[0])
    g0.smry()
    g1 = GDML.parse("$OPTICKS_PREFIX/%s.gdml" % label[1])
    g1.smry()


    lv0 = g0.find_one_volume(lvx[0])
    log.info( "lv0 %r " % lvx[0] )

    lv1 = g1.find_one_volume(lvx[1])
    log.info( "lv1 %r " % lvx[1] )


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
   
     




