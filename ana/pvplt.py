#!/usr/bin/env python

import os 
import numpy as np
import pyvista as pv
from matplotlib import collections  as mp_collections
from opticks.ana.axes import Axes

themes = ["default", "dark", "paraview", "document" ]
pv.set_plot_theme(themes[1])

GUI = not "NOGUI" in os.environ
SIZE = np.array([1280, 720])
eary_ = lambda ekey, edef:np.array( list(map(float, os.environ.get(ekey,edef).split(","))) )
efloat_ = lambda ekey, edef: float( os.environ.get(ekey,edef) )


def pvplt_simple(pl, xyz, label):
    """  
    :param pl: pyvista plotter 
    :param xyz: (n,3) shaped array of positions
    :param label: to place on plot 
    """
    pl.add_text( "pvplt_simple %s " % label, position="upper_left")
    pl.add_points( xyz, color="white" )     


def pvplt_viewpoint(pl, reset=False):
    eye = eary_("EYE",  "1,1,1.")
    look = eary_("LOOK", "0,0,0")    
    up = eary_("UP", "0,0,1")
    zoom = efloat_("ZOOM", "1")

    PARA = "PARA" in os.environ 
    print("pvplt_viewpoint reset:%d PARA:%d " % (reset, PARA))
    print(" eye  : %s " % str(eye) )
    print(" look : %s " % str(look) )
    print(" up   : %s " % str(up) )
    print(" zoom : %s " % str(zoom) )

    
    if PARA:
        pl.camera.ParallelProjectionOn()
    pass
    pl.set_focus(    look )
    pl.set_viewup(   up )
    pl.set_position( eye, reset=reset )   
    pl.camera.Zoom(zoom)


def pvplt_photon( pl, p   ):
    assert p.shape == (1,4,4)
    pos = p[:,0,:3]   
    mom = p[:,1,:3]   
    pol = p[:,2,:3]   

    pl.add_points( pos, color="magenta", point_size=16.0 )

    lmom = np.zeros( (2, 3), dtype=np.float32 )
    lmom[0] = pos[0]
    lmom[1] = pos[0] + mom[0]

    lpol = np.zeros( (2, 3), dtype=np.float32 )
    lpol[0] = pos[0]
    lpol[1] = pos[0] + pol[0]

    pl.add_lines( lmom, color="red" ) 
    pl.add_lines( lpol, color="blue" ) 



def pvplt_plotter(label="pvplt_plotter"):
    pl = pv.Plotter(window_size=SIZE*2 )  
    pl.show_grid()
    TEST = os.environ.get("TEST","")
    pl.add_text( "%s %s " % (label,TEST), position="upper_left")
    return pl 


def pvplt_arrows( pl, pos, vec, color='yellow', factor=0.15 ):
    """
     
    glyph.orient
        Use the active vectors array to orient the glyphs
    glyph.scale
        Use the active scalars to scale the glyphs
    glyph.factor
        Scale factor applied to sclaing array
    glyph.geom
        The geometry to use for the glyph

    """
    init_pl = pl == None 
    if init_pl:
        pl = pvplt_plotter(label="pvplt_arrows")   
    pass
    pos_cloud = pv.PolyData(pos)
    pos_cloud['vec'] = vec
    vec_arrows = pos_cloud.glyph(orient='vec', scale=False, factor=factor )

    pl.add_mesh(pos_cloud, render_points_as_spheres=True, show_scalar_bar=False)
    pl.add_mesh(vec_arrows, color=color, show_scalar_bar=False)


def pvplt_lines( pl, pos, vec, color='white' ):
    init_pl = pl == None 
    if init_pl:
        pl = pvplt_plotter(label="pvplt_line")   
    pass
    pos_cloud = pv.PolyData(pos)
    pos_cloud['vec'] = vec
    geom = pv.Line(pointa=(0.0, 0., 0.), pointb=(1.0, 0., 0.),)
    vec_lines = pos_cloud.glyph(orient='vec', scale=False, factor=1.0, geom=geom)
    pl.add_mesh(pos_cloud, render_points_as_spheres=True, show_scalar_bar=False)
    pl.add_mesh(vec_lines, color=color, show_scalar_bar=False)


def pvplt_add_contiguous_line_segments( pl, xpos ):
    """
    :param pl: pyvista plotter
    :param xpos: (n,3) array of positions 
    
    Adds red points at *xpos* and joins them with blue line segments using add_lines.
    This has been used only for small numbers of positions such as order less than 10 
    photon step positions.
    """
    pl.add_points( xpos, color="red" )        
    xseg = contiguous_line_segments(xpos)
    pl.add_lines( xseg, color="blue" )


def mpplt_add_contiguous_line_segments(ax, xpos, axes, linewidths=2):
    """
    :param ax: matplotlib 2D axis 
    :param xpos: (n,3) array of positions
    :param axes: (2,)  2-tuple identifying axes to pick from the where X=0, Y=1, Z=2  
    """
    assert len(axes) == 2 
    xseg = contiguous_line_segments(xpos[:,:3]).reshape(-1,2,3)  # (n,2,3) 
    xseg2D = xseg[:,:,axes]   #  (n,2,2)    same as xseg[...,axes]  see https://numpy.org/doc/stable/user/basics.indexing.html 

    lc = mp_collections.LineCollection(xseg2D, linewidths=linewidths, colors="red") 
    ax.add_collection(lc)


def ce_line_segments( ce, axes ):
    """
    :param ce: center-extent array of shape (4,)
    :param axes: tuple of length 2 picking axes from X=0 Y=1 Z=2
    :return box_lseg: box line segments of shape (4, 2, 3) 


       tl            tr = c + [e,e]
         +----------+
         |          |
         |----c-----| 
         |          |
         +----------+
       bl            br = c + [e,-e]
 
    """    
    assert len(axes) == 2 
    other_axis = Axes.OtherAxis(axes)

    c = ce[:3]
    e = ce[3]

    h = e*Axes.UnitVector(axes[0])
    v = e*Axes.UnitVector(axes[1])

    tr = c + h + v 
    bl = c - h - v 
    tl = c - h + v    
    br = c + h - v  

    box_lseg = np.empty( (4,2,3), dtype=np.float32 )
    box_lseg[0] = (tl, tr) 
    box_lseg[1] = (tr, br) 
    box_lseg[2] = (br, bl)
    box_lseg[3] = (bl, tl)
    return box_lseg 


def mpplt_ce(ax, ce, axes, linewidths=2, colors="blue"):
    assert len(axes) == 2 
    box_lseg = ce_line_segments(ce, axes)
    box_lseg_2D = box_lseg[:,:,axes]
    lc = mp_collections.LineCollection(box_lseg_2D, linewidths=linewidths, colors=colors) 
    ax.add_collection(lc)

def pvplt_ce(pl, ce, axes, color="blue"):
    assert len(axes) == 2 
    box_lseg = ce_line_segments(ce, axes)
    box_lseg_pv = box_lseg.reshape(-1,3)
    pl.add_lines( box_lseg_pv, color=color )


def contiguous_line_segments( pos ):
    """
    :param pos: (n,3) array of positions 
    :return seg: ( 2*(n-1),3) array of line segments suitable for pl.add_lines 

    Note that while the pos is assumed to represent a contiguous sequence of points
    such as photon step record positions the output line segments *seg* 
    are all independent so they could be concatenated to draw line sequences 
    for multiple photons with a single pl.add_lines

    A set of three positions (3,3):: 

        [0, 0, 0], [1, 0, 0], [1, 1, 0]

    Would give seg (2,2,3)::

         [[0,0,0],[1,0,0]],
         [[1,0,0],[1,1,0]]

    Which is reshaped to (4,3)::

         [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

    That is the form needed to represent line segments between the points
    with pl.add_lines 

    """
    assert len(pos.shape) == 2 and pos.shape[1] == 3 and pos.shape[0] > 1 
    num_seg = len(pos) - 1
    seg = np.zeros( (num_seg, 2, 3), dtype=pos.dtype )
    seg[:,0] = pos[:-1]   # first points of line segments skips the last position
    seg[:,1] = pos[1:]    # second points of line segments skipd the first position
    return seg.reshape(-1,3)
    

def test_pvplt_contiguous_line_segments():
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    seg = pvplt_contiguous_line_segments(pos)
    x_seg = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

    print(pos)
    print(seg)
    assert np.all( x_seg == seg )



def pvplt_check_transverse( mom, pol, assert_transverse=True ):
    mom_pol_transverse = np.abs(np.sum( mom*pol , axis=1 )).max() 

    if mom_pol_transverse > 1e-5:
        print("WARNING mom and pol ARE NOT TRANSVERSE mom_pol_transverse %s assert_transverse %d " % ( mom_pol_transverse , assert_transverse ))
        if assert_transverse:
            assert mom_pol_transverse < 1e-5 
        pass
    else:
        print("pvplt_check_transverse  mom_pol_transverse %s " % (mom_pol_transverse  )) 
    pass  




def pvplt_polarized( pl, pos, mom, pol, factor=0.15, assert_transverse=True ):
    """
    https://docs.pyvista.org/examples/00-load/create-point-cloud.html
    https://docs.pyvista.org/examples/01-filter/glyphs.html

    Note bizarre issue of mom arrows only in one direction appearing ?

    Managed to get them to appear using add_arrows and fiddling with mag 
    in CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.py
    """
    pvplt_check_transverse(mom, pol, assert_transverse=assert_transverse) 

    init_pl = pl == None 
    if init_pl:
        pl = pvplt_plotter(label="pvplt_polarized")   
    pass

    pos_cloud = pv.PolyData(pos)
    pos_cloud['mom'] = mom
    pos_cloud['pol'] = pol
    mom_arrows = pos_cloud.glyph(orient='mom', scale=False, factor=factor )
    pol_arrows = pos_cloud.glyph(orient='pol', scale=False, factor=factor )

    pl.add_mesh(pos_cloud, render_points_as_spheres=True, show_scalar_bar=False)
    pl.add_mesh(pol_arrows, color='lightblue', show_scalar_bar=False)
    pl.add_mesh(mom_arrows, color='red', show_scalar_bar=False)

    if init_pl:
        cp = pl.show() if GUI else None 
    else:
        cp = None 
    pass    
    return cp 


if __name__ == '__main__':
    test_pvplt_contiguous_line_segments()
pass

