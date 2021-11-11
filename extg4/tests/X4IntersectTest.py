#!/usr/bin/env python
"""



"""

import os, logging, numpy as np
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


class X4IntersectTest(object):
    CXS = os.environ.get("CXS", "pmt_solid")
    CXS_RELDIR = os.environ.get("CXS_RELDIR", "extg4/X4IntersectTest" )
    #"GeoChainSolidTest"

    FOLD = os.path.expandvars("/tmp/$USER/opticks/%s" % CXS_RELDIR )

    def __init__(self, cxs=CXS):
        base = os.path.join(self.FOLD, str(cxs))
        print("CXS : %s : loading from : %s " % (cxs,base) )
        names = os.listdir(base)
        for name in filter(lambda n:n.endswith(".npy") or n.endswith(".txt"),names):
            path = os.path.join(base, name)
            is_npy = name.endswith(".npy")
            is_txt = name.endswith(".txt")
            stem = name[:-4]
            a = np.load(path) if is_npy else list(map(str.strip,open(path).readlines())) 
            print(" %10s : %15s : %s " % (stem, str(a.shape) if is_npy else len(a), path ))  
            #globals()[stem] = a
            setattr(self, stem, a ) 
        pass



efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))

if __name__ == '__main__':
    t = X4IntersectTest()

    X,Y,Z = 0,1,2 

    ipos = t.isect[:,0,:3]
    gpos = t.gs[:,5,:3]    # last line of the transform is translation

    xlim = np.array([gpos[:,X].min(), gpos[:,X].max()])
    ylim = np.array([gpos[:,Y].min(), gpos[:,Y].max()])
    zlim = np.array([gpos[:,Z].min(), gpos[:,Z].max()])

    xx = efloatlist_("XX")
    zz = efloatlist_("ZZ")

    icol = "red"
    gcol = "grey"

    size = np.array([1280, 720])


    if mp: 
        sz = 0.1
        fig, ax = mp.subplots(figsize=size/100.)
        ax.set_aspect('equal')
        ax.scatter( ipos[:,0], ipos[:,2], s=sz, color=icol ) 
        ax.scatter( gpos[:,0], gpos[:,2], s=sz, color=gcol ) 
        for z in zz:   # ZZ horizontals 
            ax.plot( xlim, [z,z], label="z:%8.4f" % z )
        pass
        for x in xx:    # XX verticals 
            ax.plot( [x, x], zlim, label="x:%8.4f" % x )
        pass
        ax.legend()
        fig.show()
    pass

    if pv:
        #zoom = 1e-4     # black
        #zoom = 8.9e-5    # black
        zoom = 8.8e-5    # vis dot which can pull out to enlarge   ParallelScale 
        #zoom = 1e-6    # vis dot 
        #zoom = 1 

        yoffset = -50.       ## with parallel projection are rather insensitive to eye position distance
        #yoffset = -1. 
     
        up = (0,0,1)       
        look = (0,0,0)
        eye = look + np.array([ 0, yoffset, 0 ])    

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        pl.view_xz()
        pl.add_points( ipos, color=icol )
        pl.add_points( gpos, color=gcol )

        pl.camera.ParallelProjectionOn()  
        pl.camera.Zoom(zoom)
        pl.add_text("topline", position="upper_left")
        pl.add_text("botline", position="lower_left")
        pl.add_text("thirdline", position="lower_right")
        pl.set_position( eye, reset=False )
        pl.set_focus(    look )
        pl.set_viewup(   up )

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
        #cp = pl.show(screenshot=outpath)
        cp = pl.show()

        #print(cp)
        #print(pl.camera)
    pass


