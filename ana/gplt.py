#!/usr/bin/env python
"""
gplt.py : 2d debugging/presentation of solids with rotational symmetry
========================================================================

Make connection between GDML parsing and the simple 2d matplotlib plotting 
of xplt.py while avoiding the need for manual translation as done in the x018.py etc..

"""

import os, sys, argparse, logging
import numpy as np, math 
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
sys.path.insert(0, os.path.expanduser("~"))  # assumes $HOME/opticks 

from opticks.analytic.gdml import GDML, odict
from opticks.ana.torus_hyperboloid import Tor, Hyp
from opticks.ana.shape import X, SEllipsoid, STubs, STorus, SCons, SSubtractionSolid, SUnionSolid, SIntersectionSolid


class GPlot(object):
    """
    GPlot
    ------

    2d plotting small pieces of GDML defined geometry  

    """
    def __init__(self, lv, args):
        self.root = lv 
        self.args = args

    @classmethod
    def pmt_volname(cls, idx=0, pfx="NNVTMCPPMT"):
        dlv = odict()
        dlv[0] = "lMaskVirtual"
        dlv[1] = "lMask"
        dlv[2] = "_PMT_20inch_log"
        dlv[3] = "_PMT_20inch_body_log"
        dlv[4] = "_PMT_20inch_inner1_log"
        dlv[5] = "_PMT_20inch_inner2_log"
        return "%s%s" % (pfx, dlv[idx]) 

    @classmethod
    def parse_args(cls, doc):
        parser = argparse.ArgumentParser(__doc__)
        parser.add_argument( "--path", default="$OPTICKS_PREFIX/tds.gdml")

        defaults = {}
        #defaults["lvx"] = "lInnerWater"
        defaults["lvx"] = cls.pmt_volname(3)
        defaults["maxdepth"] = -1
        defaults["xlim"] = "-300,300"
        defaults["ylim"] = "-410,200"
        defaults["size"] = "8,8"
        defaults["color"] = "r,g,b,c,y,m,k" 
      
        parser.add_argument( "--lvx", default=defaults["lvx"], help="LV name prefix" )
        parser.add_argument( "--maxdepth", type=int, default=defaults["maxdepth"], help="Maximum local depth of volumes to plot" )
        parser.add_argument( "--xlim", default=defaults["xlim"], help="x limits : comma delimited string of two values" )
        parser.add_argument( "--ylim", default=defaults["ylim"], help="y limits : comma delimited string of two values" )
        parser.add_argument( "--size", default=defaults["size"], help="figure size in inches : comma delimited string of two values" )
        parser.add_argument( "--color", default=defaults["color"], help="comma delimited string of color strings" )

        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=fmt)

        fsplit_ = lambda s:map(float,s.split(",")) 
        args.xlim = fsplit_(args.xlim)
        args.ylim = fsplit_(args.ylim)
        args.size = fsplit_(args.size)
        args.color = args.color.split(",")

        return args

    def plot_r(self, lv0, ax, recurse, depth=0, **kwa):
        """
        Suspect this is assuming no offsets at pv/lv level 
        """
        pvs = lv0.physvol
        indent = "   " * depth 
        log.debug("[%2d] %s %4d %s " % (depth, indent,len(pvs),lv0.name))

        s = lv0.solid     
        color = self.args.color[lv0.local_index]
        kwa.update(color=color)

        sh = s.as_shape(**kwa)
        x = X(sh)       # X provides a place to spawn modified geometry

        for pt in x.root.patches():
            log.debug("pt %s" % pt)
            ax.add_patch(pt)
        pass

        if recurse and ( depth < self.args.maxdepth or self.args.maxdepth == -1):
            for pv in pvs:
                lv = pv.volume
                self.plot_r( lv, ax, recurse, depth+1)
            pass
        else:
            pass
        pass

    def plot(self, ax, recurse=True, **kwa):
        self.plot_r(self.root, ax, recurse=recurse, depth=0, **kwa )


    @classmethod
    def MakeFig(cls, plt, lv, args, recurse=True):
        """
        With recurse True all subvolumes are drawn onto the same canvas
        """

        fig = plt.figure(figsize=args.size)
        fig.suptitle(lv.local_prefix) 

        plt.title("gplt : %s " % lv.local_title)

        ax = fig.add_subplot(111)

        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim) 

        gp = cls( lv, args)
        gp.plot(ax, recurse=recurse)

        return fig 

    @classmethod
    def MultiFig(cls, plt, lvs, args):
        """
        Separate canvas for each LV
        """
        for lv in lvs.values():
            print(lv)
            fig = cls.MakeFig(plt, lv, args, recurse=False)
            fig.show()
        pass


    @classmethod
    def SubFig(cls, plt, lvs, args):
        """
        All volumes on one page via subplots  
        """
        ny, nx = 2, len(lvs)/2

        log.info("SubFig ny:%d nx:%d lvs:%d" % (ny,nx,len(lvs)) )

        kwa = dict()
        kwa["sharex"] = True 
        kwa["sharey"] = True 
        kwa["figsize"] = (nx*3,ny*3)
        #kwa["gridspec_kw" ] = {'hspace': 0}

        fig, axs = plt.subplots(ny, nx, **kwa )
        fig.suptitle(lvs[0].local_prefix) 

        iv = 0 
        for iy in range(ny):
            for ix in range(nx):
                if iv < len(lvs):
                    lv = lvs[iv]
                    if len(axs.shape) == 1:
                        ax = axs[iy]
                    else:
                        ax = axs[iy,ix]
                    pass
                    ax.set_title(lv.local_title)
                    ax.set_xlim(args.xlim)
                    ax.set_ylim(args.ylim) 
                    gp = cls( lv, args)
                    gp.plot(ax, recurse=False)
                pass
                iv += 1 
            pass
        pass
        return fig 




if __name__ == '__main__':

    args = GPlot.parse_args(__doc__)
    g = GDML.parse(args.path)
    g.smry()

    lv = g.find_one_volume(args.lvx)
    log.info( "lv %r " % lv )

    lvs = g.get_traversed_volumes( lv, maxdepth=args.maxdepth )

    plt.ion()

    fig = GPlot.MakeFig(plt, lv, args, recurse=True)
    fig.show()
    
    #GPlot.MultiFig(plt, lvs, args)

    fig = GPlot.SubFig(plt, lvs, args)
    fig.show()




