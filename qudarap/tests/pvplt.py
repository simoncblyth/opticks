#!/usr/bin/env python

import os 
import numpy as np
import pyvista as pv

GUI = not "NOGUI" in os.environ
SIZE = np.array([1280, 720])

def pvplt_simple(pl, xyz, label):
    """  
    :param pl: pyvista plotter 
    :param xyz: (n,3) shaped array of positions
    :param label: to place on plot 
    """
    pl.add_text( "pvplt_simple %s " % label, position="upper_left")
    pl.add_points( xyz, color="white" )     


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



def pvplt_polarized( pl, pos, mom, pol ):
    """
    https://docs.pyvista.org/examples/00-load/create-point-cloud.html
    https://docs.pyvista.org/examples/01-filter/glyphs.html
    """
    mom_pol_transverse = np.abs(np.sum( mom*pol , axis=1 )).max() 
    assert mom_pol_transverse < 1e-5 

    if pl == None:
        init_pl = True 
        pl = pv.Plotter(window_size=SIZE*2 )  
        pl.show_grid()
        TEST = os.environ["TEST"]
        pl.add_text( "pvplt_polarized %s " % TEST, position="upper_left")
    else:
        init_pl = False 
    pass

    pos_cloud = pv.PolyData(pos)
    pos_cloud['mom'] = mom
    pos_cloud['pol'] = pol
    mom_arrows = pos_cloud.glyph(orient='mom', scale=False, factor=0.15,)
    pol_arrows = pos_cloud.glyph(orient='pol', scale=False, factor=0.15,)
    pl.add_mesh(pos_cloud, render_points_as_spheres=True)
    pl.add_mesh(pol_arrows, color='lightblue')
    pl.add_mesh(mom_arrows, color='red')

    if init_pl:
        cp = pl.show() if GUI else None 
    else:
        cp = None 
    pass    
    return cp 

