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

from opticks.analytic.GDML import GDML, odict
from opticks.ana.gargs import GArgs 


class GPlot(object):
    """
    GPlot
    ------

    2d plotting small pieces of GDML defined geometry  

    """
    def __init__(self, lv, args):
        self.root = lv 
        self.args = args

    def plot(self, ax, recurse=True, **kwa):
        log.debug("kwa %s " % kwa)
        self.plot_r(self.root, ax, recurse=recurse, depth=0, **kwa )

    def plot_r(self, lv0, ax, recurse, depth, **kwa):
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

        for pt in sh.patches():
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
    def MakeFigX(cls, plt, lvs, args, recurse=True):
        """
        With recurse True all subvolumes are drawn onto the same canvas
        """
        ny = 1
        nx = len(lvs) 
        fig, axs = plt.subplots(ny, nx, **kwa )
        assert axs.shape == (nx) 

        for i in range(len(lvs)):
            lv = lvs[i]
            ax = axs[i]
            gp = cls( lv, args)
            gp.plot(ax, recurse=True)
        pass
 

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
    def SubPlotsFig(cls, plt, lvsl, args, combiZoom=False, zoomlimits=None):
        """
        :param plt:
        :param lvsl: list containing one or more lvs lists of lv

        A list of lists for lvsl is used to allow comparisons between two 
        versions of the parsed GDML.

        All volumes on one page via subplots  
        """

        if combiZoom:
            assert not zoomlimits is None 
        pass
        shorten_title_ = getattr(args, 'shorten_title_', lambda t:t) 

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
            ny, nx = 2, n_lvs//2
        pass        

        log.info("SubFig ny:%d nx:%d n_lvs:%d" % (ny,nx,n_lvs) )

        kwa = dict()
        if not combiZoom:
            kwa["sharex"] = True 
            kwa["sharey"] = True 
        pass
        kwa["figsize"] = args.size

        cx = 2
        if combiZoom:
            izz = range(2)
            fig, axs = plt.subplots(1, cx, **kwa )
        else:
            izz = range(1)
            fig, axs = plt.subplots(ny, nx, **kwa )
        pass

        suptitle = lvs[0].local_prefix if cf == False else "%s cf %s " % (lvs0[0].local_prefix, lvs1[0].local_prefix)
        fig.suptitle(suptitle, fontsize=args.suptitle_fontsize) 

        
        for iz in izz:
            iv = 0 
            for iy in range(ny):
                for ix in range(nx):
                    if iv < len(lvs):
                        
                        if combiZoom:
                            ax = axs if cx == 1 else axs[iz]
                        elif len(axs.shape) == 1:
                            ax = axs[iy]
                        elif len(axs.shape) == 2:
                            ax = axs[iy,ix]
                        pass

                        ax.set_xlim(args.xlim)
                        ax.set_ylim(args.ylim) 
                        ax.set_aspect('equal')

                        if combiZoom and iz == 1:
                            zoomlimits(ax)  
                        pass
                        
                        lv = lvs[iv]
                        title = lv.local_title if cf == False else "%s cf %s" % (shorten_title_(lvs0[iv].local_title), shorten_title_(lvs1[iv].local_title))
                      
                        if combiZoom and iv == 0: 
                            log.info(title)
                            ax.set_title(title, fontsize=10)
                        else:
                            ax.set_title(title, fontsize=10)
                        pass

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
    args = GArgs.parse(__doc__)
    lvx = args.lvname(1)

    g = GDML.parse(args.gdmlpath(0))
    g.smry()

    lv = g.find_one_volume(lvx)
    #s = lv.solid 
    #s.sub_traverse()

    log.info( "lv %r" % lv )

    lvs = g.get_traversed_volumes( lv, maxdepth=args.maxdepth )

    plt.ion()

    fig, ax = GPlot.MakeFig(plt, lv, args, recurse=True)  # all volumes together

    ax.set_aspect('equal')

    fig.show()

    combpath = args.figpath("CombinedFig")
    log.info("saving to %s " % combpath)
    fig.savefig(combpath)
    
    #axs = GPlot.MultiFig(plt, lvs, args)

if 0:
    fig, axs = GPlot.SubPlotsFig(plt, [lvs], args)
    fig.show()
    fig.savefig(args.figpath("SplitFig"))


if 0:
    pmt = lvs(-1).solid
    pmt_annotate( ax, pmt) 

    bulbneck, endtube = pmt.first, pmt.second
    bulb, neck = bulbneck.first, bulbneck.second
    neck_offset = bulbneck.position 

    neckz = neck_offset.z
    add_line( ax, [-300,neckz], [300,neckz] )

 
