#!/usr/bin/env python
"""
X4MeshTest.py : NB use the script to use the correct python
================================================================

Usage::

    x4                     # cd ~/opticks/extg4   
    x4 ; ./X4MeshTest.sh 

    GEOM=hmsk_solidMaskTail  EYE=-1,-1,0.8 ZOOM=1.5 $IPYTHON -i tests/X4MeshTest.py 

"""
import logging
import numpy as np
from opticks.ana.fold import Fold

efloat_ = lambda ekey,edef:float(os.environ.get(ekey,edef))
efloatlist_ = lambda ekey,edef:np.array(list(map(float, filter(None, os.environ.get(ekey,edef).split(",")))))

list_str_ = lambda l:str(l)[1:-1].replace(" ","")
array_str_ = lambda a:" ".join(str(a)[1:-1].split()).replace(" ",",")


try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors
    #theme = "default"
    #theme = "dark"
    #theme = "paraview"
    theme = "document"
    pv.set_plot_theme(theme)
except ImportError:
    pv = None
    hexcolors = None
pass

class X4Mesh(object):
    @classmethod
    def MakePolyData(cls, fold):
        """
        * https://docs.pyvista.org/examples/00-load/create-poly.html
        """
        tri = fold.tri.reshape(-1,3)
        faces = np.zeros( (len(tri), 4 ), dtype=np.uint32 )
        faces[:,0] = 3   # all tri as quads were split within X4Mesh.cc  
        faces[:,1:] = tri 
        surf = pv.PolyData(fold.vtx, faces)
        return surf 

    @classmethod 
    def Load(cls, path):
        fold = Fold.Load(path)
        surf = cls.MakePolyData(fold)
        return cls(fold, surf)

    def __init__(self, fold, surf):
        self.fold = fold
        self.surf = surf


class Plt(object):
    size = np.array([1280, 720])

    @classmethod
    def anno(cls, pl): 
        default_topline = "X4MeshTest.py"
        default_botline = "BOTLINE"
        default_thirdline = "THIRDLINE"

        topline = os.environ.get("TOPLINE", default_topline)
        botline = os.environ.get("BOTLINE", default_botline) 
        thirdline = os.environ.get("THIRDLINE", default_thirdline) 

        pl.add_text(topline, position="upper_left")
        pl.add_text(botline, position="lower_left")
        pl.add_text(thirdline, position="lower_right")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #default_geom = "BoxMinusOrb"
    #default_geom = "PolyconeWithMultipleRmin"
    #default_geom = "SphereWithPhiSegment"
    default_geom = "hmsk_solidMaskTail"
    geom = os.environ.get("GEOM", default_geom)

    fold = os.path.expandvars("/tmp/$USER/opticks/extg4/X4MeshTest/%s/X4Mesh" % geom ) 
    log.info("loading from fold %s " % fold) 

    eye = efloatlist_("EYE", "-1,-1,2" )
    look = efloatlist_("LOOK", "0,0,0" )
    up = efloatlist_("UP", "0,0,1" )
    zoom = efloat_("ZOOM", "1") 

    s_eye = array_str_(eye)
    s_look = array_str_(look)
    s_up = array_str_(up)
    s_zoom = str(zoom)

    log.info(" eye %s s_eye %s " % (str(eye), s_eye )) 
    log.info(" look %s s_look %s " % (str(look), s_look )) 
    log.info(" up %s s_up %s " % (str(up), s_up )) 
    log.info(" zoom %s s_zoom %s " % (str(zoom), s_zoom )) 

    os.environ["TOPLINE"] = "GEOM=%s EYE=%s ZOOM=%s ~/opticks/extg4/X4MeshTest.sh" % (geom, s_eye, s_zoom ) 
    os.environ["BOTLINE"] = fold 
    os.environ["THIRDLINE"] = "EYE=%s" % s_eye 

    mesh = X4Mesh.Load(fold)

    size = np.array([1280, 720])
    pl = pv.Plotter(window_size=size*2 )
    pl.add_mesh(mesh.surf)

    Plt.anno(pl)

    pl.set_focus(    look )
    pl.set_viewup(   up )
    pl.set_position( eye, reset=True )   ## for reset=True to succeed to auto-set the view, must do this after add_points etc.. 
    pl.camera.Zoom(zoom)

    pl.show_grid()
 
    outpath = os.path.join(fold, "pvplot.png")
    log.info("screenshot %s " % outpath)
    pl.show(screenshot=outpath)


