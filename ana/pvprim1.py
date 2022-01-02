#!/usr/bin/env python 
"""
pvprim1.py
=============

Places a cylinder on a sphere and then calulates (eye,look,up) 
viewpoint using tangential frame coordinates with origin 
at the target cylinder. 

https://docs.pyvista.org/examples/00-load/create-geometric-objects.html
"""
from collections import OrderedDict as odict 
import numpy as np
import pyvista as pv

_white = "ffffff"
_red = "ff0000"
_green = "00ff00"
_blue = "0000ff"

DTYPE = np.float64
SIZE = np.array([1280, 720])


def make_transform(r, t, p ):
    """

           -Y    Z            -T   +P
            .  /               .  /
            . /                . /
            ./                 ./
            +------ X          +------ +R
            |                 .|
            |                . |
            |               .  |
            Y             -P   +T                              



    With the below transform get units vector mapping of (X, Y, Z)  =>  (R, T, P)::

       c2s = np.array([ 
                     [ np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)  ,  0. ],
                     [ np.cos(theta)*np.cos(phi) , np.cos(theta)*np.sin(phi) , -np.sin(theta) ,  0. ], 
                     [ -np.sin(phi)              , np.cos(phi)               , 0.             ,  0. ],
                     [ 0.                        , 0.                        , 0.             ,  1. ]
                   ], dtype=DTYPE )    
                             

    (red)   +X arrows  =>  +R radial-outwards

    (green) +Y arrows  =>  +T theta-tangent (North to South direction)

    (blue)  +Z arrows  =>  +P phi-tangent (West to East direction)


    It is best to get used to that (R P T) tangential frame rather 
    than suffering the confusion of trying to rotate to something else like (P T R)
    which might initially seem more intuitive. 

     
       Z                      X
       |  Y                   |  Y
       | /                    | /
       |/                     |/
       +----- X        Z -----+

    """
    radius = r
    theta = t*np.pi
    phi = p*np.pi

    tx = radius*np.sin(theta)*np.cos(phi)  
    ty = radius*np.sin(theta)*np.sin(phi)  
    tz = radius*np.cos(theta)  

    tra = np.array([[1,   0,  0,  0],
                   [0,   1,  0,  0],
                   [0,   0,  1,  0],
                   [tx, ty, tz,  1]])

    rot = np.array([ 
                     [ np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)  ,  0. ],
                     [ np.cos(theta)*np.cos(phi) , np.cos(theta)*np.sin(phi) , -np.sin(theta) ,  0. ], 
                     [ -np.sin(phi)              , np.cos(phi)               , 0.             ,  0. ],
                     [ 0.                        , 0.                        , 0.             ,  1. ]
                   ], dtype=DTYPE )    


    tra_rot = np.array([ 
                     [ np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)  ,  0. ],
                     [ np.cos(theta)*np.cos(phi) , np.cos(theta)*np.sin(phi) , -np.sin(theta) ,  0. ], 
                     [ -np.sin(phi)              , np.cos(phi)               , 0.             ,  0. ],
                     [ tx                        , ty                        , tz             ,  1. ]
                   ], dtype=DTYPE )    



    #return np.dot(rot, tra )
    return tra_rot


class TangentialFrame(object):
    """
    Tangential frame origin is a fixed point on the surface of a sphere.
    For example a point on the X axis (like null_island on equator)::
    
        xyz :(radius,0,0)   
        theta:0.5 phi:0 
        rtpw:(0,0,0,1)     


    TODO: no extent scaling yet 
    TODO: pyvista view does not appear until interact with the view somehow
 
    """
    def __init__(self, rtp, extent=1):
        r,t,p = rtp
        radius = r
        theta = t*np.pi
        phi = p*np.pi

        x = radius*np.sin(theta)*np.cos(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(theta)
        w = 1.

        rot_tra = np.array([ 
                     [ np.sin(theta)*np.cos(phi)        , np.sin(theta)*np.sin(phi)        , np.cos(theta)         ,  0. ],
                     [ np.cos(theta)*np.cos(phi)        , np.cos(theta)*np.sin(phi)        , -np.sin(theta)        ,  0. ], 
                     [ -np.sin(phi)                     , np.cos(phi)                      , 0.                    ,  0. ],
                     [ radius*np.sin(theta)*np.cos(phi) , radius*np.sin(theta)*np.sin(phi) , radius*np.cos(theta)  ,  1. ]
                     ], dtype=DTYPE )    

        self.rot_tra = rot_tra
        self.xyzw = np.array([x,y,z,w], dtype=DTYPE)
        self.ce = np.array([x,y,z,extent], dtype=DTYPE)


    def tangential_to_conventional(self, rtpw): 
        """
        aka tangential_to_conventional
                  
                   z:p
                   | 
                   |  y:t
                   | /
                   |/ 
                   +-----x:r


        :param rtpw: tangential cartesian coordinate in frame with origin at point on surface of sphere 
                      and with tangent-to-sphere and normal-to-sphere as "theta" "phi" unit vectors  
        :return  xyzw: conventional cartesian coordinate in frame with  origin at center of sphere  
        """
        return np.dot( rtpw, self.rot_tra )

 


if __name__ == '__main__':


    pl = pv.Plotter(window_size=SIZE*2 )

    edges = False
    radius = 10. 

    sphere = pv.Sphere()   # curious choice of radius 0.5 for Sphere 
    tr_sphere = np.array([[2*radius,          0,        0, 0],
                          [0,          2*radius,        0, 0],
                          [0,                 0, 2*radius, 0],
                          [0,                 0,        0, 1]])




    rtp = odict()
    rtp["null_island"] = np.array( [radius, 0.5, 0.0] )
    rtp["midlat"] = np.array(      [radius, 0.25, 0.0] )

    #k = "null_island"
    k = "midlat"

    tf = TangentialFrame(rtp[k], extent=1.) 
    print("tf.xyzw %s " % str(tf.xyzw))

    tr_cyl = tf.rot_tra

    #view = "from_above"
    view = "from_side"

    if view == "from_above":
        look_rtpw = np.array( [0.,  0.,  0., 1.] )
        eye_rtpw  = np.array( [1.,  0.,  0., 1.] )
        up_rtpw   = np.array( [0., -1.,  0., 0.] ) 
    elif view == "from_side":
        look_rtpw = np.array( [0.,   0.,  0., 1.] )
        eye_rtpw  = np.array( [0.,   5.,  0., 1.] )
        up_rtpw   = np.array( [0.,   0.,  1., 0.] ) 
    else:
        assert 0  
    pass
    
    look_xyzw = tf.tangential_to_conventional( look_rtpw )
    eye_xyzw  = tf.tangential_to_conventional( eye_rtpw )
    up_xyzw   = tf.tangential_to_conventional( up_rtpw )
    zoom = 1. 

    print("look_xyzw : %s " % (str(look_xyzw)))     
    print("eye_xyzw : %s " % (str(eye_xyzw)))     
    print("up_xyzw : %s " % (str(up_xyzw)))     
    print("zoom : %s " % (str(zoom)))     

    sphere.transform(tr_sphere)
    pl.add_mesh(sphere, color=_blue, show_edges=edges, style="wireframe")
    pl.show_grid()

    cyl = pv.Cylinder(direction=(1,0,0))
    cyl.transform(tr_cyl.T)
    pl.add_mesh(cyl, color=_white, show_edges=True )

    for t in np.linspace(0,1,10):
        for p in np.linspace(0,2,10):
            tr = make_transform( radius, t, p )

            x_arrow = pv.Arrow(direction=(1,0,0))  # X->R
            y_arrow = pv.Arrow(direction=(0,1,0))  # Y->T
            z_arrow = pv.Arrow(direction=(0,0,1))  # Z->P

            x_arrow.transform(tr.T)  
            y_arrow.transform(tr.T)  
            z_arrow.transform(tr.T)  

            pl.add_mesh(x_arrow, color=_red , show_edges=False)
            pl.add_mesh(y_arrow, color=_green, show_edges=False)
            pl.add_mesh(z_arrow, color=_blue, show_edges=False)

            plane = pv.Plane(direction=(1,0,0))   # ends up tangential to sphere
            plane.transform(tr.T)
            pl.add_mesh(plane, color=_white, show_edges=edges)
        pass
    pass

    pl.set_focus(    look_xyzw )
    pl.set_viewup(   up_xyzw[:3] )
    pl.set_position( eye_xyzw )    # reset=True changes to get everything into frame, so its at odds with trying to carefully control the viewpoint     
    pl.camera.Zoom(zoom)
    #pl.update()

    cpos = pl.show()
    #pl.update()
    print(cpos)

