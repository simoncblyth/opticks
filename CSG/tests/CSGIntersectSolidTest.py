#!/usr/bin/env python
"""
CSGIntersectSolidTest.py
==========================


AnnulusFourBoxUnion
    notice that spurious intersects all from 2nd circle roots  
 


"""
import os, logging, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.SCenterExtentGenstep import SCenterExtentGenstep

from opticks.ana.gridspec import GridSpec, X, Y, Z
from opticks.ana.npmeta import NPMeta

efloatlist_ = lambda ekey,fallback:list(map(float, filter(None, os.environ.get(ekey,fallback).split(","))))

def eintlist_(ekey, fallback):
    slis = os.environ.get(ekey,fallback)
    if len(slis) == 0: 
        #slis = fallback
        return None
    slis = slis.split(",")
    return list(map(int, filter(None, slis)))


eint_ = lambda ekey, fallback:int(os.environ.get(ekey,fallback))


SIZE = np.array([1280, 720])

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


class Plotter(pv.Plotter):
    def __init__(self,**kwa):
        super(Plotter, self).__init__(**kwa)
 
    def add_arrows(self, cent, direction, mag=1, **kwargs):
        """Add arrows to plotting object."""
        direction = direction.copy()
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        direction[:,0] *= mag
        direction[:,1] *= mag
        direction[:,2] *= mag

        pdata = pyvista.vector_poly_data(cent, direction)
        # Create arrow object
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = wrap(glyph3D.GetOutput())

        return self.add_mesh(arrows, **kwargs)



def lines_rectangle_YX(center, halfside):
    """

          1                0
           +------+------+
           |      |      |
           |      |      |
           |- - - + - - -| 
           |      |      |
           |      |      |
           +------+------+
         2                3


         X
         |
         +-- Y

    """
    p0 = np.array( [center[0]+halfside[0], center[1]+halfside[1], center[2] ])
    p1 = np.array( [center[0]+halfside[0], center[1]-halfside[1], center[2] ])
    p2 = np.array( [center[0]-halfside[0], center[1]-halfside[1], center[2] ])
    p3 = np.array( [center[0]-halfside[0], center[1]+halfside[1], center[2] ])

    ll = np.zeros( (4,2,3),  dtype=np.float32 )

    ll[0] = p0, p1
    ll[1] = p1, p2
    ll[2] = p2, p3
    ll[3] = p3, p0

    return ll.reshape(-1,3)
    
    
 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    path = os.path.expandvars("$CFBASE/CSGIntersectSolidTest/$GEOM") 

    cegs = SCenterExtentGenstep(path)
    fold = cegs.fold


    gspos = fold.gs[:,5,:3]
    gsid =  fold.gs[:,1,3].copy().view(np.int8).reshape(-1,4)  

    dir_, t = fold.isect[:,0,:3], fold.isect[:,0,3]
    pos, sd = fold.isect[:,1,:3], fold.isect[:,1,3]

    isect_gsid = fold.isect[:,3,3].copy().view(np.int8).reshape(-1,4)    # genstep (ix,iy,iz,0) from which each intersect originated from     

    sd_cut = -1e-3
    select_spurious = sd < sd_cut 
    select_all = t > -999.

    ixiyiz = eintlist_("IXIYIZ", "0,0,0")
    iw = eint_("IW","-1")

    if not ixiyiz is None:
        ix,iy,iz = ixiyiz
        select_isect_gsid = np.logical_and( np.logical_and( isect_gsid[:,0] == ix , isect_gsid[:,1] == iy ), isect_gsid[:,2] == iz )
    else:
        select_isect_gsid = None
    pass  

    select = select_spurious 
    #select = select_isect_gsid
    #select = select_all
    #select = np.logical_and( select_isect_gsid, select_spurious )

    if iw > -1:
        # selecting a single intersect as well as single genstep  
        select_iw = isect_gsid[:,3] == iw 
        select = np.logical_and( select, select_iw )
    pass  

    ray_origin = fold.isect[:, 2, :3]
    ray_direction = fold.isect[:, 3, :3]


    s_t = t[select]
    s_pos = pos[select]
    s_isect_gsid = isect_gsid[select]
    s_dir = dir_[select]
    s_isect = fold.isect[select]

    s_ray_origin = ray_origin[select]
    s_ray_direction = ray_direction[select]

    log.info( "sd_cut %10.4g sd.min %10.4g sd.max %10.4g num select %d " % (sd_cut, sd.min(), sd.max(), len(s_pos)))


    again = 'AGAIN' in os.environ

    # save the isect only when have selected a single photon 
    if not again and len(s_isect) == 1:
        s_isect_path = "/tmp/s_isect.npy"
        log.info("save to %s " % s_isect_path )
        np.save(s_isect_path, s_isect )
    pass

    # load detailed records created when using DEBUG_RECORD 
    if again:
        isect_again = Fold.Load("/tmp/$USER/opticks/CSGQuery/intersect_again")  
        recs = isect_again.CSGRecord
    else:
        recs = []
    pass   

    look = np.array([0,0,0], dtype=np.float32) 
    eye  = np.array([0,0,-100], dtype=np.float32)
    up   = np.array([1,0,0], dtype=np.float32)


    pl = pv.Plotter(window_size=SIZE*2 ) 

    pl.add_points( gspos, color="yellow" )
    pl.add_points( pos, color="white" )
    #pl.add_arrows( pos, dir_, color="white", mag=10 )

    if len(recs) > 0:
        for rec in recs: 
            pl.add_arrows( rec[2,:3], rec[0,:3], color="pink", mag=10 )
        pass
    pass   

    if len(s_pos) > 0:
        pl.add_points( s_pos, color="red" )
        pl.add_arrows( s_pos, s_dir, color="red", mag=10 )
        #pl.add_arrows( s_ray_origin, s_ray_direction, color="blue", mag=s_t )  # arrows too big 

        ll = np.zeros( (len(s_pos), 2, 3), dtype=np.float32 )
        ll[:,0] = s_ray_origin
        ll[:,1] = s_pos  
        #pl.add_lines( ll.reshape(-1,3), color="blue" )   ## this joins up all the lines 
        pl.add_points( s_ray_origin, color="magenta", point_size=16.0 )
        for i in range(len(ll)):
            pl.add_lines( ll[i].reshape(-1,3), color="blue" )
        pass  
    else:
        log.info("no select with sd_cut %10.4g " % sd_cut)
    pass
    pl.show_grid()


    pl.set_focus(    look )
    pl.set_viewup(   up ) 
    pl.set_position( eye, reset=False )   
    pl.camera.Zoom(1)

    radius = 45. 
    arc = pv.CircularArc([0,  radius, 0], [0, radius, 0], [0, 0, 0], negative=True)
    pl.add_mesh(arc, color='cyan', line_width=1)

    llpx = lines_rectangle_YX([52., 0.,0.],[10.,11.5, 6.5]) 
    llpy = lines_rectangle_YX([0., 50.,0.],[15.,15.,  6.5])

    pl.add_lines(llpx, color='cyan', width=1 )
    pl.add_lines(llpy, color='cyan', width=1 )

    cp = pl.show()


