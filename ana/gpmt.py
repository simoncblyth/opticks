#!/usr/bin/env python
"""

https://stackoverflow.com/questions/22959698/distance-from-given-point-to-given-ellipse

"""

import os, sys, argparse, logging, textwrap
import numpy as np, math 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

specs_ = lambda s:filter(lambda s:s[0] != "#", filter(None,map(str.strip, textwrap.dedent(s).split("\n"))))

log = logging.getLogger(__name__)
sys.path.insert(0, os.path.expanduser("~"))  # assumes $HOME/opticks 

from opticks.analytic.gdml import GDML
from opticks.ana.shape import ellipse_points, circle_points
from opticks.ana.gplt import GPlot, add_line

if __name__ == '__main__':

    args = GPlot.parse_args(__doc__)

    lvx = specs_(r"""
    PMT_3inch_log
    NNVTMCPPMTlMaskVirtual
    HamamatsuR12860lMaskVirtual
    mask_PMT_20inch_vetolMaskVirtual 
    NNVTMCPPMT_PMT_20inch_log
    HamamatsuR12860_PMT_20inch_log
    """
    )
    ilv = 5

    labels = specs_(r"""
    tds_ngt
    tds_ngt_pcnk
    """
    )
    ila = 1
    label = labels[ila]


    path = "$OPTICKS_PREFIX/%s.gdml" % label
    g = GDML.parse(path)
    g.smry()

    lv = g.find_one_volume(lvx[ilv])

    if lv == None:
        log.fatal("failed to find ilv:%d lvx[ilv]:[%s] " % (ilv,lvx[ilv])) 
    assert lv

    #s = lv.solid 
    #s.sub_traverse()

    log.info( "lv %r" % lv )

    lvs = g.get_traversed_volumes( lv, maxdepth=args.maxdepth )

    plt.ion()

    fig, ax = GPlot.MakeFig(plt, lv, args, recurse=True)  # all volumes together
    fig.show()
    fig.savefig(args.pngpath("CombinedFig"))
    
    #axs = GPlot.MultiFig(plt, lvs, args)

    fig, axs = GPlot.SubPlotsFig(plt, [lvs], args)
    fig.show()
    fig.savefig(args.pngpath("SplitFig"))

    #scribble( axs[0,2] )


def scribble(ax):
    mm = 1. 
    deg = 2.*np.pi/360.

    m4_torus_r = 80. 
    m4_torus_angle = 45.*deg
    m4_r_2 = 254./2.
    m4_r_1 = (m4_r_2+m4_torus_r) - m4_torus_r*np.cos(m4_torus_angle)

    m4_h = m4_torus_r*np.sin(m4_torus_angle) + 5.0       # full height of the tube

    m4_h/2   #    tube to centerline torus offset : so torus centerline level with bottom of tube 

    neck_z = -210.*mm+m4_h/2.
    torus_z = neck_z - m4_h/2 

    torus_x = m4_r_2+m4_torus_r    # radial distance to center of torus circle      

    add_line(ax, [-300,torus_z], [300,torus_z] )       
    add_line(ax, [torus_x, -300], [torus_x, 300] )       
    
    e = ellipse_points( xy=[0,-5.], ex=254., ez=190., n=1000000 )

    #ax.scatter( e[:,0], e[:,1], marker="." )

    tc = np.array([torus_x,torus_z])
    tr = m4_torus_r  
    t = circle_points( xy=tc, tr=tr , n=100 )
    #ax.scatter( t[:,0], t[:,1], marker="." )

    e_inside_t = np.sqrt(np.sum(np.square(e-tc),1)) - tr < 0.  # points on the ellipse that are inside the torus circle 

    ax.scatter( e[e_inside_t][:,0], e[e_inside_t][:,1], marker="." ) 




