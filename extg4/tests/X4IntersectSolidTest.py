#!/usr/bin/env python
"""
X4IntersectSolidTest.py : 2D scatter plots of geometry intersect positions
============================================================================

* typically used from xxs.sh 

For debugging::

    ${IPYTHON:-ipython} -i tests/X4IntersectSolidTest.py

* can be adapted to allow comparison of intersect positions loaded from multiple input Fold 

TODO: factor out the common machinery used in this and ~/opticks/CSGOptiX/tests/CSGOptiXSimulateTest.py

"""

import os, logging, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.gridspec import GridSpec, X, Y, Z
from opticks.ana.npmeta import NPMeta

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp 
except ImportError:
    mp = None
pass
#mp = None

try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors  
    themes = ["default", "dark", "paraview", "document" ]
    pv.set_plot_theme(themes[1])
except ImportError:
    pv = None
    hexcolors = None
pass
#pv = None

efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))







class Plt(object):
    def __init__(self, folds, others):

        self.folds = folds
        self.fold0 = folds[0]
        self.others = others

        self.size = np.array([1280, 720])
        default_topline = "xxs.sh X4IntersectSolidTest.py"
        default_botline = self.folds[0].relbase    # base excluding first element
        default_thirdline = self.others[0].relbase if len(self.others) > 0 else "thirdline"

        self.topline = os.environ.get("TOPLINE", default_topline)
        self.botline = os.environ.get("BOTLINE", default_botline) 
        self.thirdline = os.environ.get("THIRDLINE", default_thirdline) 


    def pv_simple3d(self, pos):
        size = self.size

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        self.anno(pl)
        pl.add_points( pos, color="white" )    
        cp = pl.show()
        return cp

    def anno(self, pl): 
        pl.add_text(self.topline, position="upper_left")
        pl.add_text(self.botline, position="lower_left")
        pl.add_text(self.thirdline, position="lower_right")




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dirfmt = "/tmp/$USER/opticks/extg4/X4IntersectSolidTest/%(geom)s/X4Intersect" 

    GEOM_default = "hmsk_solidMaskTail"
    GEOM = os.environ.get("GEOM", GEOM_default )
    geoms = GEOM.split(",")

    colors = "red green blue cyan magenta yellow".split()

    folds = []
    others = []

    for i,geom in enumerate(geoms):

        path = os.path.expandvars( dirfmt % locals() )
        log.info(" path %s " % path )

        fold = Fold.Load(path)
        folds.append(fold)
        fold.color = colors[i]
    pass

    fold0 = folds[0]

    log.info("fold0.base %s " % fold0.base )

    gsmeta = NPMeta(fold0.gs_meta)

    grid = GridSpec(fold0.peta, gsmeta)
    is_planar = not grid.axes is None

    if is_planar:
        H,V = grid.axes     ## traditionally H,V = X,Z  but are now generalizing 
        _H,_V = grid.axlabels
    else:
        log.info("non-planar 3D grids detected" )
        H,V,_H,_V = None, None, None, None
    pass

    outdir = os.path.join(fold0.base, "figs")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pass
    savefig = True
    figname = "isect"
    print("outdir %s " % outdir)


if 1:
    ipos = fold0.isect[:,0,:3]
    gpos = fold0.gs[:,5,:3]    # last line of the transform is translation

    lim = {}
    lim[X] = np.array([gpos[:,X].min(), gpos[:,X].max()])
    lim[Y] = np.array([gpos[:,Y].min(), gpos[:,Y].max()])
    lim[Z] = np.array([gpos[:,Z].min(), gpos[:,Z].max()])

    xx = efloatlist_("XX")
    zz = efloatlist_("ZZ")
    zzd = efloatlist_("ZZD")
    CIRCLE = efloatlist_("CIRCLE") 

    icol = "red"
    other_icol = "blue"
    gcol = "grey"

    #other_offset = np.array( [10.,0.,0.] )
    other_offset = np.array( [0.,0.,0.] )


    plt = Plt(folds, others)

    if pv and not is_planar:
        ipos = fold0.isect[:,0,:3]
        plt.pv_simple3d(ipos)

    elif mp and is_planar: 
        sz = 0.1
        fig, ax = mp.subplots(figsize=plt.size/100.) # 100 dpi 
        fig.suptitle("\n".join([plt.topline,plt.botline,plt.thirdline]))

        ax.set_aspect('equal')

        ax.set_xlabel(_H)
        ax.set_ylabel(_V)

        for fold in folds:
            geom = fold.base.split("/")[-2]
            ipos = fold.isect[:,0,:3]
            gpos = fold.gs[:,5,:3]    # last line of the transform is translation
            ax.scatter( ipos[:,H], ipos[:,V], s=sz, color=fold.color, label=geom ) 
            ax.scatter( gpos[:,H], gpos[:,V], s=sz, color=gcol ) 
        pass

        for other in others:
            other_ipos = other.photons[:,0,:3] 
            ax.scatter( other_ipos[:,H]+other_offset[H], other_ipos[:,V]+other_offset[V], s=sz, color=other_icol ) 
        pass

        zlabel = True 
        xlabel = False

        if H == X and V == Z:
            for z in zz:   # ZZ horizontals 
                label = "z:%8.4f" % z if zlabel else None
                ax.plot( lim[H], [z,z], label=label )
            pass
            for z in zzd:   # ZZ horizontals 
                ax.plot( lim[H], [z,z], label=label, linestyle="dashed" )
            pass
            for x in xx:    # XX verticals 
                label = "x:%8.4f" % x if xlabel else None
                ax.plot( [x, x], lim[V], label=label )
            pass
        elif H == Z and V == X:  ## ZZ verticals 
            for z in zz:
                label = "z:%8.4f" % z if zlabel else None
                ax.plot( [z, z], lim[V], label=label )
            pass
            for z in zzd:
                label = "z:%8.4f" % z if zlabel else None
                ax.plot( [z, z], lim[V], label=label, linestyle="dashed" )
            pass
            for x in xx:    # XX horizontals
                label = "x:%8.4f" % x if xlabel else None
                ax.plot( lim[H], [x, x], label=label )
            pass
        pass


        if len(CIRCLE) > 0:
            xlim = ax.get_xlim() 
            ylim = ax.get_ylim() 

            assert len(CIRCLE) == 4 
            circle = np.array(CIRCLE) 
            circ = mp.Circle( (circle[H], circle[V]), circle[-1], color='r', clip_on=False, fill=False)
            ax.add_patch(circ)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            pass
        pass

        ax.legend(loc="upper right")
        fig.show()

        if savefig:
            outpath = os.path.join(outdir,figname+"_mpplt.png")
            print("saving %s " % outpath)
            fig.savefig(outpath)
        pass
    pass

    if pv and is_planar and 0:
        yoffset = -50.  ## with parallel projection are rather insensitive to eye position distance

        up = grid.up
        look = grid.look
        eye = grid.eye

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        for test in tests:
            geom = test.base.split("/")[-2]
            ipos = test.isect[:,0,:3]
            gpos = test.gs[:,5,:3]    # last line of the transform is translation
            pl.add_points( ipos, color=test.color )
            pl.add_points( gpos, color=gcol )
        pass

        for other in others:
            other_ipos = other.photons[:,0,:3] 
            pl.add_points( other_ipos+other_offset, color=other_icol )
        pass

        pl.add_text(plt.topline, position="upper_left")
        pl.add_text(plt.botline, position="lower_left")
        pl.add_text(plt.thirdline, position="lower_right")

        pl.camera.ParallelProjectionOn()  
        pl.set_focus(    look )
        pl.set_viewup(   up )
        pl.set_position( eye, reset=True )  # reset=False is default

        zoom = 2 
        pl.camera.Zoom(zoom)

        for z in zz:  # ZZ horizontals
            xhi = np.array( [lim[X][1], 0, z] )  # RHS
            xlo = np.array( [lim[X][0], 0, z] )  # LHS
            line = pv.Line(xlo, xhi)
            pl.add_mesh(line, color="w")
        pass
        for x in xx:    # XX verticals 
            zhi = np.array( [x, 0, lim[Z][1]] )  # TOP
            zlo = np.array( [x, 0, lim[Z][0]] )  # BOT
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


