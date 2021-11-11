#!/usr/bin/env python
"""



"""

import os, logging, numpy as np
log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)
import matplotlib.pyplot as plt 

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

X,Y,Z = 0,1,2


class X4IntersectTest(object):
    CXS = os.environ.get("CXS", "PMTSim_inner_solid_1_9")
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





if __name__ == '__main__':
    t = X4IntersectTest()

    ipos = t.isect[:,0,:3]
    gpos = t.gs[:,5,:3]    # last line of the transform is translation

    size = np.array([1280, 720])


    if 0: 
        fig, ax = plt.subplots(figsize=size/100.)
        ax.set_aspect('equal')
        ax.scatter( ipos[:,0], ipos[:,2] ) 
        fig.show()
    pass


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
    pl.add_points( ipos, color="red" )
    pl.add_points( gpos, color="white" )


    pl.camera.ParallelProjectionOn()  
    pl.camera.Zoom(zoom)
    pl.add_text("topline", position="upper_left")
    pl.add_text("botline", position="lower_left")
    pl.add_text("thirdline", position="lower_right")
    pl.set_position( eye, reset=False )
    pl.set_focus(    look )
    pl.set_viewup(   up )
    cp = pl.show()

    #print(cp)
    #print(pl.camera)



