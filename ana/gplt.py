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
import matplotlib.lines as mlines

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
    def pmt_volname(cls, idx=0, pfx0="NNVTMCPPMT", pfx1="_PMT_20inch"):  
        """
        PMT volumes changed very recently, old GDML needs pfx1="_PMT_20inch" new GDML pfx1=""
        """
        dlv = odict()

        dlv[0] = "lMaskVirtual"   # gone ?
        dlv[1] = "lMask"          # gone ?
        dlv[2] = "_log" 
        dlv[3] = "_body_log" 
        dlv[4] = "_inner1_log"
        dlv[5] = "_inner2_log" 

        return "%s%s%s" % (pfx0, pfx1, dlv[idx]) 

    @classmethod
    def parse_args(cls, doc):
        parser = argparse.ArgumentParser(__doc__)
        parser.add_argument( "--path", default="$OPTICKS_PREFIX/tds.gdml")

        defaults = {}
        #defaults["lvx"] = "lInnerWater"
        defaults["lvx"] = cls.pmt_volname(2)
        defaults["maxdepth"] = -1    
        defaults["xlim"] = "-300,300"
        defaults["ylim"] = "-410,200"
        defaults["size"] = "8,8"
        defaults["color"] = "r,g,b,c,y,m,k" 
        defaults["figdir"] = "/tmp/fig"       
        #defaults["figpfx"] = "TorusNeck"       
        defaults["figpfx"] = "PolyconeNeck"       

        parser.add_argument( "--lvx", default=defaults["lvx"], help="LV name prefix" )
        parser.add_argument( "--maxdepth", type=int, default=defaults["maxdepth"], help="Maximum local depth of volumes to plot, 0 for just root, -1 for no limit" )
        parser.add_argument( "--xlim", default=defaults["xlim"], help="x limits : comma delimited string of two values" )
        parser.add_argument( "--ylim", default=defaults["ylim"], help="y limits : comma delimited string of two values" )
        parser.add_argument( "--size", default=defaults["size"], help="figure size in inches : comma delimited string of two values" )
        parser.add_argument( "--color", default=defaults["color"], help="comma delimited string of color strings" )
        parser.add_argument( "--figdir", default=defaults["figdir"], help="directory path in which to save PNG figures" )
        parser.add_argument( "--figpfx", default=defaults["figpfx"], help="prefix for PNG filename" )

        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=fmt)

        fsplit_ = lambda s:map(float,s.split(",")) 
        args.xlim = fsplit_(args.xlim)
        args.ylim = fsplit_(args.ylim)
        args.size = fsplit_(args.size)
        args.color = args.color.split(",")

        if not os.path.isdir(args.figdir):
            os.makedirs(args.figdir)
        pass 

        args.pngpath = lambda _:os.path.join(args.figdir, "%s%s.png" % (args.figpfx,_))
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
        log.debug("kwa %s " % kwa)
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

        return fig, ax 

    @classmethod
    def MultiFig(cls, plt, lvs, args):
        """
        Separate canvas for each LV
        """
        axs = []
        for lv in lvs.values():
            print(lv)
            fig, ax = cls.MakeFig(plt, lv, args, recurse=False)
            fig.show()
            axs.append(ax)
        pass
        return axs


    @classmethod
    def SubPlotsFig(cls, plt, lvsl, args):
        """
        :param plt:
        :param lvsl: list containing one or more lvs lists of lv

        A list of lists for lvsl is used to allow comparisons between two 
        versions of the parsed GDML.

        All volumes on one page via subplots  
        """
        if len(lvsl) == 1:
            cf = False
            lvs = lvsl[0]
        elif len(lvsl) == 2:
            cf = True
            lvs0 = lvsl[0]
            lvs1 = lvsl[1]
            assert len(lvs0) == len(lvs1), (len(lvs0), len(lvs1))
            lvs = lvs0
        pass 

        n_lvs = len(lvs) 

        if n_lvs == 3:
            ny, nx = 2, 2
        else:
            ny, nx = 2, n_lvs/2
        


        log.info("SubFig ny:%d nx:%d n_lvs:%d" % (ny,nx,n_lvs) )

        kwa = dict()
        kwa["sharex"] = True 
        kwa["sharey"] = True 
        #kwa["figsize"] = (nx*3,ny*3)
        kwa["figsize"] = args.size
        #kwa["gridspec_kw" ] = {'hspace': 0}

        fig, axs = plt.subplots(ny, nx, **kwa )

        suptitle = lvs[0].local_prefix if cf == False else "%s cf %s " % (lvs0[0].local_prefix, lvs1[0].local_prefix)
        fig.suptitle(suptitle) 

        iv = 0 
        for iy in range(ny):
            for ix in range(nx):
                if iv < len(lvs):
                    if len(axs.shape) == 1:
                        ax = axs[iy]
                    else:
                        ax = axs[iy,ix]
                    pass
                    ax.set_xlim(args.xlim)
                    ax.set_ylim(args.ylim) 
                    
                    lv = lvs[iv]
                    title = lv.local_title if cf == False else "%s cf %s" % (lvs0[iv].local_title, lvs1[iv].local_title)

                    ax.set_title(title)

                    if cf == False:
                        gp = cls( lv, args)
                        gp.plot(ax, recurse=False)
                    else:
                        gp0 = cls( lvs0[iv], args)
                        gp0.plot(ax, recurse=False)
                        gp1 = cls( lvs1[iv], args)
                        gp1.plot(ax, recurse=False, linestyle="dotted")
                    pass 
                pass
                iv += 1 
            pass
        pass
        return fig, axs




def add_line(ax, p0, p1, **kwa):
    x0,y0 = p0
    x1,y1 = p1
    l = mlines.Line2D([x0, x1], [y0, y1], *kwa )
    ax.add_line(l)


def pmt_annotate( ax, pmt):
    """
    """
    bulbneck, endtube = pmt.first, pmt.second

    bulb = bulbneck.first
    neck = bulbneck.second 

    assert bulb.is_primitive, bulb
    assert bulb.__class__.__name__ == 'Ellipsoid', bulb

    # position applies offset to second boolean constituent 
    ztub = bulbneck.position.z  # neck offset 
    add_line(ax, [-300,ztub], [300,ztub])
    add_line( ax, [0,-400], [0,400] )

    if not neck.__class__.__name__ == 'Polycone':
        tube = neck.first 
        torus = neck.second
        ztor = bulbneck.position.z + neck.position.z  # absolute neck offset 

        rtor = torus.rtor
        add_line(ax, [rtor, -400], [rtor, 400])
        add_line(ax, [-rtor, -400], [-rtor, 400])

        add_line( ax, [-rtor,ztor], [0,0] )
        add_line( ax, [rtor,ztor], [0,0] )
        add_line(ax, [-300,ztor], [300,ztor] ) 
    pass




if __name__ == '__main__':

    args = GPlot.parse_args(__doc__)
    g = GDML.parse(args.path)
    g.smry()


    #lvx = "NNVTMCPPMT_PMT_20inch_log"
    #lvx = "NNVTMCPPMT_log"
    lvx = "HamamatsuR12860_PMT_20inch_body_log" 

    lv = g.find_one_volume(lvx)
    s = lv.solid 
    s.sub_traverse()

    log.info( "lv %r " % lv )

    lvs = g.get_traversed_volumes( lv, maxdepth=args.maxdepth )

    plt.ion()

    fig, ax = GPlot.MakeFig(plt, lv, args, recurse=True)  # all volumes together
    fig.show()
    fig.savefig(args.pngpath("CombinedFig"))
    
    #axs = GPlot.MultiFig(plt, lvs, args)

    fig, axs = GPlot.SubPlotsFig(plt, [lvs], args)
    fig.show()
    fig.savefig(args.pngpath("SplitFig"))



if 0:
    pmt = lvs(-1).solid
    pmt_annotate( ax, pmt) 

    bulbneck, endtube = pmt.first, pmt.second
    bulb, neck = bulbneck.first, bulbneck.second
    neck_offset = bulbneck.position 

    neckz = neck_offset.z
    add_line( ax, [-300,neckz], [300,neckz] )

 
