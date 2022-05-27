#!/usr/bin/env python
"""
tests/CSGOptiXSimtraceTest.py
==============================

* see notes/issues/simtrace-shakedown.rst


See also:

csg/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh



This is allows interactive visualization of workstation 
generated intersect data fphoton.npy on remote machines such as 
user laptops that support pyvista. 

issue : TR inversion of pv vs mp
-----------------------------------

* PR cross section matches, TR is inverted in T direction
 
pyvista GUI keys
----------------------

* https://docs.pyvista.org/api/plotting/plotting.html

* to zoom out/in : slide two fingers up/down on trackpad. 
* to pan : hold down shift and one finger tap-lock, then move finger around  


Too many items in the legend
-----------------------------

When not using MASK=pos the legend may be filled with feature item lines 
that are not visible in the frame 


FramePhotons vs Photons
---------------------------

Using frame photons is a trick to effectively see results 
from many more photons that have to pay the costs for transfers etc.. 
Frame photons lodge photons onto a frame of pixels limiting 
the maximumm number of photons to handle. 

ISEL allows plotting of a selection of feature values only, picked by descending frequency index
-------------------------------------------------------------------------------------------------

::

    cx ; ./cxs_Hama.sh  grab
    cx ; ./cxs_Hama.sh  ana 

Old instructions, not recently exercised::

    cx ; ./cxs_grab.sh   ## NO LONGER USED ?

    ISEL=0,1         ./cxs.sh    # ISEL=0,1 picks the 2 most frequent feature values (eg boundaries when FEAT=bnd)
    ISEL=0,1,2,3,4   ./cxs.sh 

    ISEL=Hama        ./cxs.sh    # select boundaries via strings in the bndnames, assuming FEAT=bnd
    ISEL=NNVT        ./cxs.sh 
    ISEL=Pyrex       ./cxs.sh 
    ISEL=Pyrex,Water ./cxs.sh 


"""
import os, sys, logging, numpy as np
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)
log = logging.getLogger(__name__)

SIZE = np.array([1280, 720])
GUI = not "NOGUI" in os.environ
MP =  not "NOMP" in os.environ 
PV =  not "NOPV" in os.environ 
LEGEND =  not "NOLEGEND" in os.environ # when not MASK=pos legend often too many lines, so can switch it off 
PVGRID = "PVGRID" in os.environ
SIMPLE = "SIMPLE" in os.environ
MASK = os.environ.get("MASK", "pos")
FEAT = os.environ.get("FEAT", "pid" )  
ALLOWED_MASK = ("pos", "t", "non" )
assert MASK in ALLOWED_MASK, "MASK %s is not in ALLOWED_MASK list %s " % (MASK, str(ALLOWED_MASK))
GSPLOT = int(os.environ.get("GSPLOT", "0"))


from opticks.CSG.CSGFoundry import CSGFoundry 
from opticks.ana.p import *       # including cf loaded from CFBASE
from opticks.ana.fold import Fold
from opticks.ana.feature import SimtraceFeatures
from opticks.ana.simtrace_positions import SimtracePositions
from opticks.ana.framegensteps import FrameGensteps
from opticks.ana.npmeta import NPMeta
from opticks.sysrap.sframe import sframe , X, Y, Z

import matplotlib
if GUI == False:
    log.info("set pdf backend as GUI False")
    matplotlib.use("agg")
pass

if MP:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        mp = None
    pass
else:
    mp = None
pass

if PV:
    try:
        import pyvista as pv
        themes = ["default", "dark", "paraview", "document" ]
        pv.set_plot_theme(themes[1])
    except ImportError:
        pv = None
    pass
else:
    pv = None
pass

if GUI == False:
    log.info("disabling pv as GUI False")
    pv = None
pass


def pvplt_simple(xyz, label):
    """
    :param xyz: (n,3) shaped array of positions
    :param label: to place on plot 

    KEEP THIS SIMPLE : FOR DEBUGGING WHEN LESS BELLS AND WHISTLES IS AN ADVANTAGE
    """
    pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
    pl.add_text( "pvplt_simple %s " % label, position="upper_left")
    pl.add_points( xyz, color="white" )        
    pl.show_grid()
    cp = pl.show() if GUI else None
    return cp


class SimtracePlot(object):
    def __init__(self, pl, feat, gs, frame, pos, outdir ):
        """
        :param pl: pyvista plotter instance, can be None
        :param feat: Feature instance 
        :param gs: FrameGensteps instance
        :param frame: sframe instance
        :param pos: Positions instance
        :param outdir: str

        ## hmm regarding annotation, what should come from remote and what local ?

        XX,YY,ZZ 
           lists of ordinates for drawing lines parallel to axes

        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        pass
        self.pl = pl
        self.feat = feat
        self.gs = gs
        self.frame = frame
        self.pos = pos
        self.outdir = outdir 
        self.pl = None

        topline = os.environ.get("TOPLINE", "CSGOptiXSimtraceTest.py:PH")
        botline = os.environ.get("BOTLINE", "cxs") 
        note = os.environ.get("NOTE", "") 
        note1 = os.environ.get("NOTE1", "") 

        self.topline = topline 
        self.botline = botline 
        self.note = note 
        self.note1 = note1 

        efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))

        aa = {} 
        aa[X] = efloatlist_("XX")
        aa[Y] = efloatlist_("YY")
        aa[Z] = efloatlist_("ZZ")

        self.aa = aa
        self.sz = float(os.environ.get("SZ","1.0"))
        self.zoom = float(os.environ.get("ZOOM","3.0"))

        log.info(" aa[X] %s " % str(self.aa[X]))
        log.info(" aa[Y] %s " % str(self.aa[Y]))
        log.info(" aa[Z] %s " % str(self.aa[Z]))
 
    def outpath_(self, stem="positions", ptype="pvplt"):
        sisel = self.feat.sisel
        return os.path.join(self.outdir,"%s_%s_%s.png" % (stem, ptype, self.feat.name)) 

    def positions_mpplt(self):
        axes = self.frame.axes   
        if len(axes) == 2:
            self.positions_mpplt_2D(legend=LEGEND, gsplot=GSPLOT)
        else:
            log.info("mp skip 3D plotting as PV is so much better at that")
        pass

    def positions_mpplt_2D(self, legend=True, gsplot=0):
        """
        (H,V) are the plotting axes 
        (X,Y,Z) = (0,1,2) correspond to absolute axes which can be mapped to plotting axes in various ways 
        
        when Z is vertical lines of constant Z appear horizontal 
        when Z is horizontal lines of constant Z appear vertical 
        """
        upos = self.pos.upos  # may have mask applied 
        ugsc = self.gs.ugsc
        lim = self.gs.lim

        H,V = self.frame.axes       # traditionally H,V = X,Z  but now generalized
        _H,_V = self.frame.axlabels

        log.info(" frame.axes H:%s V:%s " % (_H, _V))  

        feat = self.feat
        sz = self.sz
        print("positions_mpplt feat.name %s " % feat.name )

        xlim = lim[X]
        ylim = lim[Y]
        zlim = lim[Z]

        igs = slice(None) if len(ugsc) > 1 else 0

        title = [self.topline, self.botline, self.frame.thirdline]

        fig, ax = mp.subplots(figsize=SIZE/100.)  # mpl uses dpi 100
        fig.suptitle("\n".join(title))
              
        note = self.note
        note1 = self.note1
        if len(note) > 0:
             mp.text(0.01, 0.99, note, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        pass
        if len(note1) > 0:
             mp.text(0.01, 0.95, note1, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        pass

        # loop over unique values of the feature 
        for idesc in range(feat.unum):
            uval, selector, label, color, skip, msg = feat(idesc)
            if skip: continue
            pos = upos[selector]    
            ## hmm any masking needs be applied to both upos and selector ?
            ## alternatively could apply the mask early and use that from the Feature machinery 

            ax.scatter( pos[:,H], pos[:,V], label=label, color=color, s=sz )
        pass

        log.info(" xlim[0] %8.4f xlim[1] %8.4f " % (xlim[0], xlim[1]) )
        log.info(" ylim[0] %8.4f ylim[1] %8.4f " % (ylim[0], ylim[1]) )
        log.info(" zlim[0] %8.4f zlim[1] %8.4f " % (zlim[0], zlim[1]) )

        self.lines_plt(ax, None)

        label = "gs_center XZ"

        if gsplot > 0:
            ax.scatter( ugsc[igs, H], ugsc[igs,V], label=None, s=sz )
        pass

        ax.set_aspect('equal')
        ax.set_xlim( lim[H] )
        ax.set_ylim( lim[V] ) 
        ax.set_xlabel(_H)
        ax.set_ylabel(_V)

        if legend:
            ax.legend(loc="upper right", markerscale=4)
            ptype = "mpplt"  
        else:
            ptype = "mpnky"  
        pass
        if GUI:
            fig.show()
        pass 

        outpath = self.outpath_("positions",ptype )
        print(outpath)
        fig.savefig(outpath)


    def lines_plt(self, ax, pl):
        """
        Draws axis parallel line segments in matplotlib and pyvista.
        The segments extend across the genstep grid limits.
        Lines to draw are configured using comma delimited value lists 
        in envvars XX, YY, ZZ

        :param ax: matplotlib axis
        :param pl: pyvista plot 


               +----------------------+
               |                      |
               |                      |
               +----------------------+      
               |                      |   
               +----------------------+      
               |                      |
               |                      |   V=Z
               +----------------------+

                 H=X

        """
        H,V = self.frame.axes    
        hlim = self.gs.lim[H]
        vlim = self.gs.lim[V]

        for i in [X,Y,Z]: 
            aa = self.aa[i]
            if len(aa) > 0:
                for a in aa:
                    if V == i:
                        if not ax is None:
                            ax.plot( hlim, [a,a] )
                        elif not pl is None:
                            pass
                            lo = np.array( [0, 0, 0])
                            hi = np.array( [0, 0, 0])

                            lo[H] = hlim[0] 
                            hi[H] = hlim[1] 
                            lo[i] = a 
                            hi[i] = a 

                            hline = pv.Line(lo, hi)
                            pl.add_mesh(hline, color="w")
                        pass
                    elif H == i:
                        if not ax is None:
                            ax.plot( [a,a], vlim )
                        elif not pl is None:
                            pass
                            lo = np.array( [0, 0, 0])
                            hi = np.array( [0, 0, 0])

                            lo[V] = vlim[0] 
                            hi[V] = vlim[1] 
                            lo[i] = a 
                            hi[i] = a 

                            vline = pv.Line(lo, hi)
                            pl.add_mesh(vline, color="w")
                        pass
                    else:
                        pass
                    pass
                pass
            pass
        pass



    def positions_pvplt(self):
        axes = self.frame.axes   
        if len(axes) == 2:
            self.positions_pvplt_2D()
        else:
            self.positions_pvplt_3D()
        pass

    @classmethod
    def MakePVPlotter(cls):
        log.info("MakePVPlotter")
        pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
        return pl 

    def get_pv_plotter(self):
        if self.pl is None:  
            pl = self.MakePVPlotter()
            self.pl = pl
        else:
            pl = self.pl
            log.info("using preexisting plotter")
        pass
        return pl 


    def positions_pvplt_3D(self):
        """
        Could try to reconstruct solid surface from the point cloud of intersects 
        https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyDataFilters.reconstruct_surface.html#pyvista.PolyDataFilters.reconstruct_surface
        """
        pass
        pl = self.get_pv_plotter()

        feat = self.feat 
        upos = self.pos.upos   ## typically local frame 

        log.info("feat.unum %d " % feat.unum)

        for idesc in range(feat.unum):
            uval, selector, label, color, skip, msg = feat(idesc)
            if skip: continue
            pos = upos[selector] 
            print(msg)
            pl.add_points( pos[:,:3], color=color, point_size=10  )
        pass
        pl.enable_eye_dome_lighting()  
        ##
        ## improves depth peception for point cloud, especially from a distance
        ## https://www.kitware.com/eye-dome-lighting-a-non-photorealistic-shading-technique/
        ##
        pl.show_grid()
        cp = pl.show() if GUI else None
        return cp


    def positions_pvplt_2D(self):
        """
        * actually better to use set_position reset=True after adding points to auto get into ballpark 

        * previously always starts really zoomed in, requiring two-finger upping to see the intersects
        * following hint from https://github.com/pyvista/pyvista/issues/863 now set an adhoc zoom factor
 
        Positioning the eye with a simple global frame y-offset causes distortion 
        and apparent untrue overlaps due to the tilt of the geometry.
        Need to apply the central transform to the gaze vector to get straight on view.

        https://docs.pyvista.org/api/core/_autosummary/pyvista.Camera.zoom.html?highlight=zoom

        In perspective mode, decrease the view angle by the specified factor.

        In parallel mode, decrease the parallel scale by the specified factor.
        A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.       

        """

        lim = self.gs.lim
        ugsc = self.gs.ugsc

        xlim = lim[X]
        ylim = lim[Y]
        zlim = lim[Z]

        H,V = self.frame.axes      ## traditionally H,V = X,Z  but are now generalizing 
 
        upos = self.pos.upos

        feat = self.feat 
        zoom = self.zoom
        look = self.frame.look if self.pos.local else self.frame.ce[:3]
        eye = look + self.frame.off
        up = self.frame.up


        pl = self.get_pv_plotter()

        #pl.view_xz()   ## TODO: see if view_xz is doing anything when subsequently set_focus/viewup/position 

        pl.camera.ParallelProjectionOn()  
        pl.add_text(self.topline, position="upper_left")
        pl.add_text(self.botline, position="lower_left")
        pl.add_text(self.frame.thirdline, position="lower_right")

        print("positions_pvplt feat.name %s " % feat.name )

        for idesc in range(feat.unum):
            uval, selector, label, color, skip, msg = feat(idesc)
            if skip: continue
            pos = upos[selector] 
            print(msg)
            pl.add_points( pos[:,:3], color=color )
        pass

        showgrid = len(self.frame.axes) == 2 # too obscuring with 3D
        if showgrid:
            pl.add_points( ugsc[:,:3], color="white" )   # genstep grid
        pass   

        ## the lines need reworking 

        self.lines_plt(None, pl)

        # TODO: use gridspec.pv_compose 

        pl.set_focus(    look )
        pl.set_viewup(   up )
        pl.set_position( eye, reset=True )   ## for reset=True to succeed to auto-set the view, must do this after add_points etc.. 
        pl.camera.Zoom(2)

        if PVGRID:
            pl.show_grid()
        pass

    def positions_pvplt_show(self):
        outpath = self.outpath_("positions","pvplt")
        print(outpath)
        cp = self.pl.show(screenshot=outpath)
        return cp


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    t = Fold.Load(symbol="t"); 
    frame = sframe.Load(t.base, "sframe.npy")  # TODO: automate specialized loading in Fold
    simtrace = t.photon # TODO:rename array to simtrace.npy as contents nothing like photon 

    x = Fold.Load("$CFBASE/CSGOptiXSimTest", symbol="x")


    SimtracePositions.Check(simtrace)

    local = True 

    gs = FrameGensteps(t.genstep, frame, local=local)  ## get gs positions in target frame

    pos = SimtracePositions(simtrace, gs, frame, local=local, mask=MASK )
    upos = pos.upos

    if SIMPLE:
        pvplt_simple(pos.gpos[:,:3], "pos.gpos[:,:3]" )
        pvplt_simple(pos.lpos[:,:3], "pos.lpos[:,:3]" )
        raise Exception("SIMPLE done")
    pass

    pf = SimtraceFeatures(pos, cf, featname=FEAT ) 

    pl = SimtracePlot.MakePVPlotter()

    plt = SimtracePlot(pl, pf.feat, gs, frame, pos, outdir=os.path.join(t.base, "figs") )

    if not mp is None:
        plt.positions_mpplt()
    pass

    if not pv is None:
        plt.positions_pvplt()
        plt.positions_pvplt_show()
    pass

pass
