#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.eget import efloatlist_, elookce_, elook_epsilon_, eint_
from opticks.sysrap.sframe import sframe , X, Y, Z

SIZE = np.array([1280, 720])   ## SIZE*2 [2560, 1440]
LEGEND =  not "NOLEGEND" in os.environ # when not MASK=pos legend often too many lines, so can switch it off 
GSPLOT = eint_("GSPLOT", "0")



MP =  not "NOMP" in os.environ
if MP: 
    try:
        import matplotlib.pyplot as mp
        from matplotlib.patches import Circle, Rectangle, Ellipse
    except ImportError:
        mp = None
    pass
else:
    mp = None
pass


PV =  not "NOPV" in os.environ
PVGRID = not "NOPVGRID" in os.environ
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

from opticks.ana.pvplt import *



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

        self.look = efloatlist_("LOOK", "0,0,0")
        self.look_ce = elookce_(extent=10)


        epsilon = self.frame.propagate_epsilon
        if epsilon == 0.: epsilon = 0.05 
        self.look_epsilon = elook_epsilon_(epsilon)


        aa = {} 
        aa[X] = efloatlist_("XX")
        aa[Y] = efloatlist_("YY")
        aa[Z] = efloatlist_("ZZ")

        self.aa = aa
        self.sz = float(os.environ.get("SZ","1.0"))

        self.ellipse0 = efloatlist_("ELLIPSE0", "0,0,0,0")      #  elw,elh,elx,ely,ela,ez0,ez1
        self.ellipse1 = efloatlist_("ELLIPSE1", "0,0,0,0")      #  
        self.rectangle0 = efloatlist_("RECTANGLE0", "0,0,0,0")  #  hx,hy,cx,cy,al,dy
        self.rectangle1 = efloatlist_("RECTANGLE1", "0,0,0,0")  # 

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
              
        self.ax = ax  # TODO: arrange to pass this in 
        self.fig = fig 

        note = self.note
        note1 = self.note1
        if len(note) > 0:
             mp.text(0.01, 0.99, note, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        pass
        if len(note1) > 0:
             mp.text(0.01, 0.95, note1, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        pass

        if len(self.ellipse0) > 0 and self.ellipse0[0] > 0.:
            mpplt_add_ellipse(ax, self.ellipse0)
        pass
        if len(self.ellipse1) > 0 and self.ellipse1[0] > 0.:
            mpplt_add_ellipse(ax, self.ellipse1)
        pass
        if len(self.rectangle0) > 0 and self.rectangle0[0] > 0.:
            mpplt_add_rectangle(ax, self.rectangle0)
        pass
        if len(self.rectangle1) > 0 and self.rectangle1[0] > 0.:
            mpplt_add_rectangle(ax, self.rectangle1)
        pass

        # loop over unique values of the feature 
        for idesc in range(feat.unum):
            uval, selector, label, color, skip, msg = feat(idesc)
            if skip: continue
            pos = upos[selector]    
            ## hmm any masking needs be applied to both upos and selector ?
            ## alternatively could apply the mask early and use that from the Feature machinery 
            log.info(" feat.unum %d idesc %d label %s color %s msg %s " % (feat.unum, idesc, label, color, msg ))
            ax.scatter( pos[:,H], pos[:,V], label=label, color=color, s=sz )
        pass

        log.info(" xlim[0] %8.4f xlim[1] %8.4f " % (xlim[0], xlim[1]) )
        log.info(" ylim[0] %8.4f ylim[1] %8.4f " % (ylim[0], ylim[1]) )
        log.info(" zlim[0] %8.4f zlim[1] %8.4f " % (zlim[0], zlim[1]) )

        mpplt_parallel_lines(ax, self.gs.lim, self.aa, self.frame.axes, self.look, linestyle="dashed" ) 

        if hasattr(self, 'x_lpos'):
            x_lpos = self.x_lpos
            ax.scatter( x_lpos[:,H], x_lpos[:,V], label="x_lpos", s=10 )
            mpplt_add_contiguous_line_segments(ax, x_lpos, axes=self.frame.axes, linewidths=2)
        pass

        if hasattr(self, 't_spos'):
            t_spos = self.t_spos
            ax.scatter( t_spos[:,H], t_spos[:,V], label="t_spos", s=5 )
        pass

        if hasattr(self, 'simtrace_selection'):
            sts = self.simtrace_selection
            mpplt_simtrace_selection_line(ax, sts, axes=self.frame.axes, linewidths=2)
        pass

        if not self.look_ce is None:
            mpplt_ce_multiple(ax, self.look_ce, axes=self.frame.axes)
        pass

        if not self.look_epsilon is None:
            mpplt_ce(ax, self.look_epsilon, axes=self.frame.axes, colors="yellow" ) 
        pass


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
        ## improves depth peception for point cloud, especially from a distance
        ## https://www.kitware.com/eye-dome-lighting-a-non-photorealistic-shading-technique/
        pl.show_grid()


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

        pl = self.get_pv_plotter()

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

        if hasattr(self, 'x_lpos'):
            pvplt_add_contiguous_line_segments(pl, self.x_lpos[:,:3], point_size=25 )
        pass

        if hasattr(self, 'simtrace_selection'):
            sts = self.simtrace_selection
            pvplt_simtrace_selection_line(pl, sts)
        pass

        if not self.look_ce is None:
            pvplt_ce_multiple(pl, self.look_ce, axes=self.frame.axes)
        pass

        if not self.look_epsilon is None:
            pvplt_ce(pl, self.look_epsilon, axes=self.frame.axes, color="yellow" ) 
        pass

        show_genstep_grid = len(self.frame.axes) == 2 # too obscuring with 3D
        if show_genstep_grid and PVGRID:
            pl.add_points( ugsc[:,:3], color="white" ) 
        pass   

        pvplt_parallel_lines(pl, self.gs.lim, self.aa, self.frame.axes, self.look ) 

        self.frame.pv_compose(pl, local=True) 


        outpath = self.outpath_("positions","pvplt")
        print(outpath)
        cp = pl.show(screenshot=outpath)

        #img = pl.screenshot(outpath, return_img=True)
        #print("img.shape %s " % str(img.shape) )
        #self.img = img 

        return cp 


    def mp_show(self):
        """
        Fatal Python error: Segmentation fault 
        """
        if hasattr(self, 'img'):
            mp.imshow(self.img)
            mp.show()               
        pass
          


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    

