#!/usr/bin/env python 
"""
tangential.py
==============

Calculates positions of points on tangent planes to sphere 
using a tangential frame. 

See also:

* ana/spherical.py 
* tangential.cc
* https://mathworld.wolfram.com/SphericalCoordinates.html
* https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

"""

import numpy as np
from collections import OrderedDict as odict
from opticks.ana.spherical import Spherical 

import pyvista as pv
pv.set_plot_theme("dark")

_white = "ffffff"
_red = "ff0000"
_green = "00ff00"
_blue = "0000ff"

DTYPE = np.float64
FLOAT = [float, np.float64, np.float32]

SIZE = np.array([1280, 720])

def spherical_to_cartesian( rtp ):
    """
       | x |       |  r*sin(theta)*cos(phi)  |
       |   |       |                         |
       | y |   =   |  r*sin(theta)*sin(phi)  | 
       |   |       |                         |
       | z |       |      r*cos(theta)       |
    """
    
    r,t,p = rtp 

    if type(t) in FLOAT and type(p) in FLOAT:
        num = 1
    else:
        assert len(t) == len(p)
        num = len(t)
    pass

    radius = r 
    theta = t*np.pi
    phi = p*np.pi

    x = radius*np.sin(theta)*np.cos(phi)  
    y = radius*np.sin(theta)*np.sin(phi)  
    z = radius*np.cos(theta)  
    w = DTYPE(1.)

    xyzw = np.zeros([num, 4], dtype=DTYPE) 
    xyzw[:,0] = x.ravel()
    xyzw[:,1] = y.ravel()
    xyzw[:,2] = z.ravel()
    xyzw[:,3] = w.ravel()

    return xyzw 

def cartesian_to_spherical( xyzw ):
    """
    """
    x,y,z,w = xyzw

    num_x = 1 if type(x) in FLOAT else len(x)
    num_y = 1 if type(y) in FLOAT else len(y)
    num_z = 1 if type(z) in FLOAT else len(z)
    num_w = 1 if type(w) in FLOAT else len(w)

    num = num_x*num_y*num_z*num_w
    assert num_w == 1

    r = np.sqrt( x*x + y*y + z*z ) 
    t = np.arccos(z/r)
    p = np.arctan(y/x)  

    rtp = np.zeros( [num,3], dtype=DTYPE )
    rtp[:,0] = r 
    rtp[:,1] = t 
    rtp[:,2] = p 

    return rtp


def spherical_to_cartesian_unit_vectors( r, theta, phi ):
    """
    Cartesian unit vectors (xu yu zu) in terms of spherical unit vectors related by 
    the inverse of the above transform which is its transpose::


          | xu |     |   sin(theta)cos(phi)    cos(theta)cos(phi)      -sin(phi)    |  | ru | 
          |    |     |                                                              |  |    |
          | yu | =   |  sin(theta)sin(phi)    cos(theta)sin(phi)        cos(phi)    |  | tu |
          |    |     |                                                              |  |    |
          | zu |     |   cos(theta)               -sin(theta)              0        |  | pu |

    """
    pass

def cartesian_to_spherical_unit_vectors( r, theta, phi ):
    """
    Spherical unit vectors (ru tu pu) related to cartesian unit vectors (xu yu zu)
    via orthogonal rotation matrix::

          | ru |     |   sin(theta)cos(phi)    sin(theta)sin(phi)      cos(theta)   |  | xu | 
          |    |     |                                                              |  |    |
          | tu | =   |  cos(theta)cos(phi)    cos(theta)sin(phi)     -sin(theta)    |  | yu |
          |    |     |                                                              |  |    |
          | pu |     |  -sin(phi)                 cos(phi)              0           |  | zu |

    """
    c2s = np.zeros( [4,4], dtype=DTYPE )
    c2s[0,0] = np.sin(theta)*np.cos(phi)   ; c2s[0,1] = np.sin(theta)*np.sin(phi)   ; c2s[0,2] = np.cos(theta)    ; c2s[0,3] = 0. 
    c2s[1,0] = np.cos(theta)*np.cos(phi)   ; c2s[1,1] = np.cos(theta)*np.sin(phi)   ; c2s[1,2] = -np.sin(theta)   ; c2s[1,3] = 0.
    c2s[2,0] = -np.sin(phi)                ; c2s[2,1] = np.cos(phi)                 ; c2s[2,2] =  0.              ; c2s[2,3] = 0.
    c2s[3,0] = 0.                          ; c2s[3,1] = 0.                          ; c2s[3,2] =  0.              ; c2s[3,3] = 1.
    return c2s 


class Tangential(object):
    """
    Consider coordinate systems describing a point on a sphere:

    1. spherical (r,t,p) with origin at center of sphere
    2. cartesian (x,y,z,1) with origin at center of sphere
    3. cartesian using 
 
       * origin : fixed point (r,t,p) on the surface of the sphere 
       * unit vectors : normal-to-sphere and a conventional choice of "theta" and "phi" tangents-to-sphere 
         at the fixed point (r,t,p) [see spherical.py for a demonstration of such frames]
 
    What are the 4x4 matrices to transform frames 2 -> 3 and vice-versa ? 
    They will be some combination of the below rotation and translation matrices or their inverses::

          | ru |     |   sin(theta)cos(phi)    sin(theta)sin(phi)      cos(theta)        0  |  | xu | 
          |    |     |                                                                      |  |    |
          | tu |     |  cos(theta)cos(phi)    cos(theta)sin(phi)     -sin(theta)         0  |  | yu |
          |    |     |                                                                      |  |    |
          | pu |     |  -sin(phi)                 cos(phi)              0                0  |  | zu |
          |    |     |                                                                      |  |    | 
          |    |     |   0                         0                    0                1  |  |    |           

     Translation from center to the fixed point

          | ru |     |   1                     0                       0                 0  |  | xu | 
          |    |     |                                                                      |  |    |
          | tu |     |   0                     1                       0                 0  |  | yu |
          |    |     |                                                                      |  |    |
          | pu |     |   0                     0                       1                 0  |  | zu |
          |    |     |                                                                      |  |    | 
          |    |     |   r sin(theta)cos(phi)    r sin(theta)sin(phi)    r cos(theta)    1  |  |    |  


    """

    def __init__(self, rtp):
        r,t,p = rtp 
        theta = t*np.pi
        phi = p*np.pi 

        rot = np.zeros( [4,4], dtype=DTYPE )
        iro = np.zeros( [4,4], dtype=DTYPE )
        tra = np.zeros( [4,4], dtype=DTYPE )
        itr = np.zeros( [4,4], dtype=DTYPE )

        rot[0,0] = np.sin(theta)*np.cos(phi)    ; rot[0,1] = np.sin(theta)*np.sin(phi)    ; rot[0,2] = np.cos(theta)     ; rot[0,3] = 0. 
        rot[1,0] = np.cos(theta)*np.cos(phi)    ; rot[1,1] = np.cos(theta)*np.sin(phi)    ; rot[1,2] = -np.sin(theta)    ; rot[1,3] = 0.
        rot[2,0] = -np.sin(phi)                 ; rot[2,1] = np.cos(phi)                  ; rot[2,2] =  0.               ; rot[2,3] = 0.
        rot[3,0] = 0.                           ; rot[3,1] = 0.                           ; rot[3,2] =  0.               ; rot[3,3] = 1.
        iro = rot.T

        tra[0,0] = 1.                           ; tra[0,1] = 0.                           ; tra[0,2] = 0.                ; tra[0,3] = 0. 
        tra[1,0] = 0.                           ; tra[1,1] = 1.                           ; tra[1,2] = 0.                ; tra[1,3] = 0.
        tra[2,0] = 0.                           ; tra[2,1] = 0.                           ; tra[2,2] = 1.                ; tra[2,3] = 0.
        tra[3,0] = r*np.sin(theta)*np.cos(phi)  ; tra[3,1] =  r*np.sin(theta)*np.sin(phi) ; tra[3,2] = r*np.cos(theta)   ; tra[3,3] = 1.

        itr[0,0] = 1.                           ; itr[0,1] = 0.                           ; itr[0,2] = 0.                ; itr[0,3] = 0. 
        itr[1,0] = 0.                           ; itr[1,1] = 1.                           ; itr[1,2] = 0.                ; itr[1,3] = 0.
        itr[2,0] = 0.                           ; itr[2,1] = 0.                           ; itr[2,2] = 1.                ; itr[2,3] = 0.
        itr[3,0] = -r*np.sin(theta)*np.cos(phi) ; itr[3,1] = -r*np.sin(theta)*np.sin(phi) ; itr[3,2] = -r*np.cos(theta)  ; itr[3,3] = 1.

        itr_rot = np.dot(itr, rot)
        iro_tra = np.dot(iro, tra)
        rot_tra = np.dot(rot, tra)

        self.rot = rot
        self.iro = iro
        self.tra = tra
        self.itr = itr
        self.itr_rot = itr_rot  
        self.iro_tra = iro_tra  
        self.rot_tra = rot_tra  

    def conventional_to_tangential(self, xyzw): 
        """
        :param xyzw: conventional cartesian coordinate in frame with  origin at center of sphere  
        :return rtpw: tangential cartesian coordinate in frame with origin at point on surface of sphere 
                      and with tangent-to-sphere and normal-to-sphere as "theta" "phi" unit vectors  
        """
        return np.dot( xyzw, self.itr_rot )

    def tangential_to_conventional(self, rtpw): 
        """
                  
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
        #return np.dot( rtpw, self.iro_tra )
        return np.dot( rtpw, self.rot_tra )

    def get_plane_rtpw(self, side=10, num=20 ):
        v_t = np.linspace( -side, side, num )  
        v_p = np.linspace( -side, side, num )  
        t, p = np.meshgrid(v_t, v_p)
        r = np.zeros_like(t)
        w = np.ones_like(t)
        _rtpw = np.dstack( (r, t, p, w)) 
        return _rtpw 
 
    def get_plane_xyzw(self, side=10, num=20):
        _rtpw = self.get_plane_rtpw(side=side, num=num )
        _xyzw = self.tangential_to_conventional( _rtpw ) 
        return _xyzw

    def __repr__(self):
        return "\n".join(map(str, ["itr", self.itr, "rot", self.rot, "tra", self.tra, "itr_rot", self.itr_rot, "iro_tra", self.iro_tra ]))

    def pvplot(self, pl, color=_white, side=10, num=20 ):
        _xyzw = self.get_plane_xyzw(side=side, num=num)
        pos =  _xyzw[:,:,:3].reshape(-1,3)   
        pl.add_points( pos,  color=color )


if __name__ == '__main__':

    rtp = odict()
    xyzw = odict()
    xyzw2 = odict()
    rtpw = odict()
    ta = odict()

    radius = 20. 

    #rtp["center"]           = np.array( [0., 0., 0.],       dtype=DTYPE )
    rtp["north_pole"]       = np.array( [radius, 0,   0.],  dtype=DTYPE )
    rtp["null_island"]      = np.array( [radius, 0.5, 0.],  dtype=DTYPE )
    rtp["mid_island"]       = np.array( [radius, 0.5, 0.5], dtype=DTYPE )
    rtp["anti-null_island"] = np.array( [radius, 0.5, 1.],  dtype=DTYPE )
    rtp["south_pole"]       = np.array( [radius, 1,   0.],  dtype=DTYPE )
    rtp["midlat"]           = np.array( [radius, 0.25, 0.], dtype=DTYPE )
    rtp["midlat2"]          = np.array( [radius, 0.25, 0.5], dtype=DTYPE )

    for k in rtp: xyzw[k] = spherical_to_cartesian( rtp[k] )
    for k in rtp: ta[k] = Tangential( rtp[k] )
    for k in rtp: rtpw[k] = ta[k].conventional_to_tangential(xyzw[k])
    for k in rtp: xyzw2[k] = ta[k].tangential_to_conventional(rtpw[k])

    fmt = "%20s %30s %30s %30s %30s"
    print(fmt % ("name", "rtp", "xyzw", "rtpw", "xyzw2"))

    for k in rtp: print(fmt % (k, rtp[k], xyzw[k], rtpw[k], xyzw2[k] ))

    all_rtp = np.vstack(rtp.values())    
    all_xyzw = spherical_to_cartesian( all_rtp.T )

    sg = Spherical.Grid(radius=radius, n_theta=24, n_phi=24 )


    sp = pv.Sphere(radius=radius)


    pl = pv.Plotter(window_size=SIZE*2 )
    pl.show_grid()

    sg.pvplot(pl, mag=2.5)

    pl.add_mesh(sp)

    #ta["midlat2"].pvplot(pl, color="red", side=10, num=20 )

    #ta["north_pole"].pvplot(pl, color="blue" )
    #ta["south_pole"].pvplot(pl, color="yellow" )
    #ta["null_island"].pvplot(pl, color="cyan" )
    #ta["anti-null_island"].pvplot(pl, color="magenta" )


    look = np.array( [0,0,0])
    up = np.array( [0,0,1])
    eye =  3.0*radius*np.array([1.,1.,1.])/np.sqrt(3.)

    pl.set_focus(    look )
    pl.set_viewup(   up )
    pl.set_position( eye, reset=True ) 

    pl.camera.Zoom(2.0)

    outpath = "/tmp/tangential.png"
    cp = pl.show(screenshot=outpath)



