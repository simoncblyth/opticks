#!/usr/bin/env python
"""
X4IntersectSolidTest.py : 2D scatter plots of geometry intersect positions
============================================================================

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
pv = None

efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))

X,Y,Z = 0,1,2

_axes = {}
_axes[X] = "X"
_axes[Y] = "Y"
_axes[Z] = "Z"

class GridSpec(object):
    def __init__(self, peta):

        ix0,ix1,iy0,iy1 = peta[0,0].view(np.int32)
        iz0,iz1,photons_per_genstep,zero = peta[0,1].view(np.int32)

        ce = tuple(peta[0,2])
        sce = ("%7.2f" * 4 ) % ce

        assert photons_per_genstep > 0
        assert zero == 0
        nx = (ix1 - ix0)//2
        ny = (iy1 - iy0)//2
        nz = (iz1 - iz0)//2

        axes = self.determine_planar_axes(nx, ny, nz)

        self.peta = peta 
        self.axes = axes
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ce = ce


    def determine_planar_axes(self, nx, ny, nz):
        """
        :param nx:
        :param nx:
        :param nx:

        """
        if nx == 0 and ny > 0 and nz > 0:
            ny_over_nz = float(ny)/float(nz)
            axes = (Y,Z) if nx_over_nz > 1 else (Z,Y)
        elif nx > 0 and ny == 0 and nz > 0:
            nx_over_nz = float(nx)/float(nz)
            axes = (X,Z) if nx_over_nz > 1 else (Z,X)
        elif nx > 0 and ny > 0 and nz == 0:
            nx_over_ny = float(nx)/float(ny)
            axes = (X,Y) if nx_over_ny > 1 else (Y,X)
        else:
            axes = None
        pass
        if axes is None:
            msg = "non-planar grid in use : will need 3D handling "
        else:
            msg = "planar grid in use : axes %s %s " % ( _axes[axes[0]], _axes[axes[1]]) 
        pass
        print(" nx %d ny %d nz %d : %s " % (nx, ny, nz, msg))
        print("axes %s " % str(axes))
        return axes



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    CXS_RELDIR = os.environ.get("CXS_RELDIR", "extg4/X4IntersectTest" ) 
    CXS_OTHER_RELDIR = os.environ.get("CXS_OTHER_RELDIR", "CSGOptiX/CSGOptiXSimulateTest" ) 

    GEOM = os.environ.get("GEOM", "pmt_solid")
    geoms = GEOM.split(",")

    colors = "red green blue cyan magenta yellow".split()

    tests = []
    others = []
    for i,geom in enumerate(geoms):
        test = Fold.Load("/tmp/$USER/opticks",CXS_RELDIR,geom, "X4Intersect", globals=True, globals_prefix=geom )
        tests.append(test)
        test.color = colors[i]
        #other = Fold.Load("/tmp/$USER/opticks",CXS_OTHER_RELDIR,geom, globals=True, globals_prefix="other_" + geom )
        #others.append(other)
    pass

    test = tests[0]
    gridspec = GridSpec(test.peta)

    if gridspec.axes is None:
        log.fatal("only planar grids are handled by this script, not 3D ones" )
        assert 0 
    pass

    H,V = gridspec.axes     ## traditionally H,V = X,Z  but are now generalizing 

    _H = _axes[H]
    _V = _axes[V]


    outdir = os.path.join(tests[0].base, "figs")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pass
    savefig = True
    figname = "isect"
    print("outdir %s " % outdir)


if 1:
    default_topline = "xxs.sh X4IntersectSolidTest.py"
    default_botline = tests[0].relbase    # base excluding first element
    default_thirdline = others[0].relbase if len(others) > 0 else "thirdline"

    topline = os.environ.get("TOPLINE", default_topline)
    botline = os.environ.get("BOTLINE", default_botline) 
    thirdline = os.environ.get("THIRDLINE", default_thirdline) 

    ipos = tests[0].isect[:,0,:3]
    gpos = tests[0].gs[:,5,:3]    # last line of the transform is translation

    lim = {}
    lim[X] = np.array([gpos[:,X].min(), gpos[:,X].max()])
    lim[Y] = np.array([gpos[:,Y].min(), gpos[:,Y].max()])
    lim[Z] = np.array([gpos[:,Z].min(), gpos[:,Z].max()])

    xx = efloatlist_("XX")
    zz = efloatlist_("ZZ")
    zzd = efloatlist_("ZZD")

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

        ax.set_xlabel(_H)
        ax.set_ylabel(_V)

        for test in tests:
            geom = test.base.split("/")[-2]
            ipos = test.isect[:,0,:3]
            gpos = test.gs[:,5,:3]    # last line of the transform is translation
            ax.scatter( ipos[:,H], ipos[:,V], s=sz, color=test.color, label=geom ) 
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


        if H == X and V == Z:
            up = (0,0,1)       
        elif H == Z and V == X:
            up = (-1,0,0)       
        else:
            assert 0
        pass


        look = (0,0,0)
        eye = look + np.array([ 0, yoffset, 0 ])    

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        pl.view_xz()
        pl.camera.ParallelProjectionOn()  

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

        pl.add_text(topline, position="upper_left")
        pl.add_text(botline, position="lower_left")
        pl.add_text(thirdline, position="lower_right")
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


