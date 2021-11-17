#!/usr/bin/env python
"""
X4IntersectVolumeTest.py : 2D scatter plots of geometry intersect positions
============================================================================

* typically used from xxs.sh 

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

X,Y,Z = 0,1,2

efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    reldir = os.environ.get("CXS_RELDIR", "extg4/X4IntersectVolumeTest" ) 
    geom = os.environ.get("GEOM", "body_phys")

    colors = "red green blue cyan magenta yellow pink orange purple lightgreen".split()
    gcol = "grey"

    basedir = os.path.expandvars(os.path.join("/tmp/$USER/opticks",reldir, geom ))
    transforms = np.load(os.path.join(basedir, "transforms.npy"))
    transforms_meta = np.loadtxt( os.path.join(basedir, "transforms_meta.txt"), dtype=np.object ) 

    figsdir = os.path.join(basedir, "figs")
    if not os.path.isdir(figsdir):
        os.makedirs(figsdir)
    pass
    savefig = True
    figname = "isect"
    print("figsdir %s " % figsdir)

    topline = "X4IntersectVolumeTest.py"
    botline = "%s/%s " % (reldir, geom)
    thirdline = "thirdline"

    print(basedir)
    print(transforms)
    print(transforms_meta)

    isects = {}
    for soname in transforms_meta:
        isects[soname] = Fold.Load(basedir, soname, "X4Intersect")
    pass

    size = np.array([1280, 720])
    H,V = Z,X

    if mp: 
        sz = 3
        fig, ax = mp.subplots(figsize=size/100.) # 100 dpi 
        fig.suptitle("\n".join([topline,botline,thirdline]))
        ax.set_aspect('equal')
       
        soname0 = transforms_meta[0]
        isect0 = isects[soname0]
        gpos = isect0.gs[:,5,:3]    # last line of the transform is translation
        ax.scatter( gpos[:,H], gpos[:,V], s=sz, color=gcol ) 

        for i, soname in enumerate(transforms_meta):
            isect = isects[soname]
            tran = np.float32(transforms[i])
            ipos = isect.isect[:,0,:3] + tran[3,:3]
            color = colors[ i % len(colors)]
            label = str(soname[1:])   # seems labels starting "_" have special meaning 
            label = label.replace("solid","s")

            ax.scatter( ipos[:,H], ipos[:,V], s=sz, color=color, label=label ) 
        pass
        ax.legend(loc="lower left",  markerscale=4)
        fig.show()

        if savefig:
            figpath = os.path.join(figsdir,figname+"_mpplt.png")
            print("saving %s " % figpath)
            fig.savefig(figpath)
        pass 
    pass


