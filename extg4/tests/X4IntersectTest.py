#!/usr/bin/env python
"""
X4IntersectTest.py : 2D scatter plots of geometry intersect positions
========================================================================

* typically used from xxs.sh 
* provides comparison of intersect positions loaded from two input Fold 

Formerly had to fiddle around varying the pyvista zoom (often needing very small values 1e-8) 
to make geometry visible, but now find that using set_position with reset=True manages 
to automatically get into ballpark from which more reasonable zoom values can fine tune.

Observed ipos stuck at genstep origin positions causing unexpected
coloring of the genstep points, presumably from rays that miss. 
Confirmed this by observation that gentstep points inside solids
never change color as it is impossible to miss from inside. 

TODO: select on flags to avoid the miscoloring 
"""

import os, logging, numpy as np
from opticks.ana.fold import Fold

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp 
except ImportError:
    mp = None
pass

try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors  
    #theme = "default"
    theme = "dark"
    #theme = "paraview"
    #theme = "document"
    pv.set_plot_theme(theme)
except ImportError:
    pv = None
    hexcolors = None
pass

#mp = None
#pv = None

X,Y,Z = 0,1,2

efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))

if __name__ == '__main__':

    GEOM = os.environ.get("GEOM", "pmt_solid")
    CXS_RELDIR = os.environ.get("CXS_RELDIR", "extg4/X4IntersectTest" ) 
    CXS_OTHER_RELDIR = os.environ.get("CXS_OTHER_RELDIR", "CSGOptiX/CSGOptiXSimulateTest" ) 

    test = Fold.Load("/tmp/$USER/opticks",CXS_RELDIR,GEOM, globals=True, globals_prefix="a_" )
    other = Fold.Load("/tmp/$USER/opticks",CXS_OTHER_RELDIR,GEOM, globals=True, globals_prefix="b_" )

    outdir = os.path.join(test.base, "figs")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pass
    savefig = True
    figname = "isect"
    print("outdir %s " % outdir)

    default_topline = "xxs.sh X4IntersectTest.py"
    default_botline = test.relbase    # base excluding first element
    default_thirdline = other.relbase if not other is None else "thirdline"

    topline = os.environ.get("TOPLINE", default_topline)
    botline = os.environ.get("BOTLINE", default_botline) 
    thirdline = os.environ.get("THIRDLINE", default_thirdline) 

    X,Y,Z = 0,1,2 

    ipos = test.isect[:,0,:3]
    gpos = test.gs[:,5,:3]    # last line of the transform is translation
    other_ipos = other.photons[:,0,:3] if not other is None else None  
 
    xlim = np.array([gpos[:,X].min(), gpos[:,X].max()])
    ylim = np.array([gpos[:,Y].min(), gpos[:,Y].max()])
    zlim = np.array([gpos[:,Z].min(), gpos[:,Z].max()])

    xx = efloatlist_("XX")
    zz = efloatlist_("ZZ")

    icol = "red"
    other_icol = "blue"
    gcol = "grey"

    #other_offset = np.array( [10.,0.,0.] )
    other_offset = np.array( [0.,0.,0.] )
    size = np.array([1280, 720])

    if mp: 
        sz = 0.1
        fig, ax = mp.subplots(figsize=size/100.) # 100 dpi 
        fig.suptitle("\n".join([topline,botline,thirdline]))

        ax.set_aspect('equal')
        ax.scatter( ipos[:,0], ipos[:,2], s=sz, color=icol ) 
        ax.scatter( gpos[:,0], gpos[:,2], s=sz, color=gcol ) 

        if not other is None:
            ax.scatter( other_ipos[:,0]+other_offset[0], other_ipos[:,2]+other_offset[2], s=sz, color=other_icol ) 
        pass

        for z in zz:   # ZZ horizontals 
            ax.plot( xlim, [z,z], label="z:%8.4f" % z )
        pass
        for x in xx:    # XX verticals 
            ax.plot( [x, x], zlim, label="x:%8.4f" % x )
        pass

        ax.legend(loc="upper right")
        fig.show()

        if savefig:
            outpath = os.path.join(outdir,figname+"_mpplt.png")
            print("saving %s " % outpath)
            fig.savefig(outpath)
        pass
    pass

    if pv:
        yoffset = -50.  ## with parallel projection are rather insensitive to eye position distance
        up = (0,0,1)       
        look = (0,0,0)
        eye = look + np.array([ 0, yoffset, 0 ])    

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        pl.view_xz()
        pl.camera.ParallelProjectionOn()  

        pl.add_points( ipos, color=icol )
        pl.add_points( gpos, color=gcol )
        if not other is None:
            pl.add_points( other_ipos+other_offset, color=other_icol )
        pass

        pl.add_text(topline, position="upper_left")
        pl.add_text(botline, position="lower_left")
        pl.add_text(thirdline, position="lower_right")
        pl.set_focus(    look )
        pl.set_viewup(   up )
        pl.set_position( eye, reset=True )  # reset=False is default

        zoom = 2 
        pl.camera.Zoom(zoom)

        for z in zz:  # ZZ horizontals
            xhi = np.array( [xlim[1], 0, z] )  # RHS
            xlo = np.array( [xlim[0], 0, z] )  # LHS
            line = pv.Line(xlo, xhi)
            pl.add_mesh(line, color="w")
        pass
        for x in xx:    # XX verticals 
            zhi = np.array( [x, 0, zlim[1]] )  # TOP
            zlo = np.array( [x, 0, zlim[0]] )  # BOT
            line = pv.Line(zlo, zhi)
            pl.add_mesh(line, color="w")
        pass

        if savefig:
            outpath = os.path.join(outdir, figname+"_pvplt.png")
            print("saving %s " % outpath)
            cp = pl.show(screenshot=outpath)
        else:
            cp = pl.show()
        pass
        #print(cp)
        #print(pl.camera)
    pass


