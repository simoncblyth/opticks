#!/usr/bin/env python

import os 
import numpy as np
import pyvista as pv
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

    Note bizarre issue of arrows only in one direction appearing ?

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

