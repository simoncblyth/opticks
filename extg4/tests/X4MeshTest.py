#!/usr/bin/env python
"""
X4MeshTest.py : NB use the script to use the correct python
================================================================

Usage::

    x4                     # cd ~/opticks/extg4   
    x4 ; ./X4MeshTest.sh 

"""
import logging, numpy 
from opticks.ana.fold import Fold

efloatlist_ = lambda ekey,edef:list(map(float, filter(None, os.environ.get(ekey,edef).split(","))))

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
    default_geom = "PolyconeWithMultipleRmin"

    geom = os.environ.get("GEOM", default_geom)
    fold = os.path.expandvars("/tmp/$USER/opticks/extg4/X4IntersectTest/%s/X4Mesh" % geom ) 
    cpos = efloatlist_("CPOS", "-1,-1,2" )
    s_cpos = str(cpos)[1:-1].replace(" ","")



    os.environ["TOPLINE"] = "GEOM=%s CPOS=%s ~/opticks/extg4/X4MeshTest.sh" % (geom, s_cpos ) 
    os.environ["BOTLINE"] = fold 
    os.environ["THIRDLINE"] = "CPOS=%s" % s_cpos 

    mesh = X4Mesh.Load(fold)

    #mesh.surf.plot(cpos=cpos)

    size = np.array([1280, 720])
    pl = pv.Plotter(window_size=size*2 )
    pl.add_mesh(mesh.surf)

    Plt.anno(pl)
    pl.show_grid()
 
    outpath = os.path.join(fold, "pvplot.png")
    log.info("screenshot %s " % outpath)
    pl.show(cpos=cpos, screenshot=outpath)




