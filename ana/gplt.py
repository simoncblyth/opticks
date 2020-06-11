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
        self.nvol = 0 
        self.vol = odict()
        self.traverse()

    @classmethod
    def parse_args(cls, doc):
        parser = argparse.ArgumentParser(__doc__)
        parser.add_argument( "--path", default="$OPTICKS_PREFIX/tds.gdml")

        #dlv = "lInnerWater"
        dlv = "NNVTMCPPMTlMaskVirtual"
        #dlv = "NNVTMCPPMTlMask"
        #dlv = "NNVTMCPPMT_PMT_20inch_log"
        #dlv = "NNVTMCPPMT_PMT_20inch_inner1_log"

        defaults = {}
        defaults["lvx"] = dlv
        defaults["xlim"] = "-300,300"
        defaults["ylim"] = "-410,200"
        defaults["size"] = "8,8"
      
        parser.add_argument( "--lvx", default=defaults["lvx"], help="LV name prefix" )
        parser.add_argument( "--xlim", default=defaults["xlim"], help="x limits : comma delimited string of two values" )
        parser.add_argument( "--ylim", default=defaults["ylim"], help="y limits : comma delimited string of two values" )
        parser.add_argument( "--size", default=defaults["size"], help="figure size in inches : comma delimited string of two values" )

        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=fmt)

        fsplit_ = lambda s:map(float,s.split(",")) 
        args.xlim = fsplit_(args.xlim)
        args.ylim = fsplit_(args.ylim)
        args.size = fsplit_(args.size)

        return args

    def traverse_r(self, lv0, depth=0):
        pvs = lv0.physvol
        indent = "   " * depth 
        print("[%2d] %s %4d %s " % (depth, indent,len(pvs),lv0.name))
        pass
        self.vol[self.nvol] = lv0
        self.nvol += 1 

        for pv in pvs:
            lv = pv.volume
            self.traverse_r( lv, depth+1)
        pass

    def traverse(self):
        self.traverse_r(self.root, 0)
        log.info(" nvol:%d " % self.nvol)

    def plot_r(self, lv0, ax, depth=0):
        """
        Suspect this is assuming no offsets at pv/lv level 
        """
        pvs = lv0.physvol
        indent = "   " * depth 
        log.debug("[%2d] %s %4d %s " % (depth, indent,len(pvs),lv0.name))

        s = lv0.solid     
        sh = s.as_shape()
        x = X(sh)       # X provides a place to spawn modified geometry

        for pt in x.root.patches():
            log.debug("pt %s" % pt)
            ax.add_patch(pt)
        pass
        for pv in pvs:
            lv = pv.volume
            self.plot_r( lv, ax, depth+1)
        pass

    def plot(self, ax):
        self.plot_r(self.root, ax, 0)

    def combined_fig(self, plt):

        args = self.args

        log.info("size %s " % args.size)

        fig = plt.figure(figsize=args.size)
        plt.title("gplt : %s " % self.root.name)

        ax = fig.add_subplot(111)

        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)    # from -350 to -400 : got bigger ?

        self.plot(ax)

        return fig 

    def split_fig(self, plt):

        args = self.args

        for i in range(self.nvol):
            lv = self.vol[i]
            print(lv)
        pass



if __name__ == '__main__':

    args = GPlot.parse_args(__doc__)
    g = GDML.parse(args.path)
    g.smry()

    lv = g.find_one_volume(args.lvx)

    gp = GPlot(lv, args)

    plt.ion()
    fig = gp.combined_fig(plt)
    fig.show()


    gp.split_fig(plt)



