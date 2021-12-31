#!/usr/bin/env python 
"""
spherical.py
=============

Draws points on a sphere together with a (r,phi,theta) derivative vectors
which are normals and tangents to the sphere and form orthogonal 
frames at every point on the sphere (other than perhaps the poles). 

                         . 

                            .
                     
                              .     pvec (blue)   increasing phi : eg around equator or other lines of latitude, from West to East  
                                   /
                               .  /
                                 / 
      +                         +-------->
    center                      |         rvec (red)   radial normal vector
                               .|
                                |
                             . tvec (green) increasing theta : eg down lines of longitude : from North to South 
                                 
                            .
                        
                         ,

See also:

* ana/spherical_0.py ana/tangential.py 
* https://mathworld.wolfram.com/SphericalCoordinates.html
* https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

"""

import numpy as np
import pyvista as pv

_white = "ffffff"
_red = "ff0000"
_green = "00ff00"
_blue = "0000ff"

DTYPE = np.float64
SIZE = np.array([1280, 720])

class Spherical(object):
    @classmethod
    def Grid(cls, radius=20., n_theta=50, n_phi=50):
        v_theta = np.linspace(0,1,n_theta, dtype=DTYPE )   
        v_phi = np.linspace(0,2,n_phi, dtype=DTYPE) 
        num = len(v_theta)*len(v_phi) 
        theta, phi = np.meshgrid(v_theta, v_phi)
        print("v_theta.shape %s theta.shape %s " % (str(v_theta.shape),str(theta.shape)) )
        print("v_phi.shape %s phi.shape %s " % (str(v_phi.shape),str(phi.shape)))
        sph = cls((radius, theta, phi))
        return sph

    @classmethod
    def One(cls, radius=20., theta=0.5, phi=0.0):
        sph = cls((radius, theta, phi))
        return sph

    def __init__(self, rtp ):
        r,t,p = rtp
          
        num_r = 1 if type(r) is float else len(r)
        num_t = 1 if type(t) is float else len(t)
        num_p = 1 if type(p) is float else len(p)
        num = num_r*num_t*num_p

        assert num_r == 1

        radius = r 
        theta = t*np.pi
        phi = p*np.pi

        # derivative with radius of definition of spherical coordinates : direction of increasing radius : normal vector
        x = np.cos(phi)*np.sin(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(theta)

        one = np.sqrt(x*x + y*y + z*z)

        rvec = np.zeros( (num,4),  dtype=DTYPE )
        pvec = np.zeros( (num,4),  dtype=DTYPE )
        tvec = np.zeros( (num,4),  dtype=DTYPE )

        rvec[:,0] = x.ravel()
        rvec[:,1] = y.ravel() 
        rvec[:,2] = z.ravel() 
        rvec[:,3] = 0.

        # derivative with theta  : direction of increasing theta : theta tangent vector
        tvec[:,0] =  (np.cos(phi)*np.cos(theta)).ravel()
        tvec[:,1] =  (np.sin(phi)*np.cos(theta)).ravel()
        tvec[:,2] = -np.sin(theta).ravel() 
        tvec[:,3] = 0.

        # derivative with phi : direction of increasing phi : phi tangent vector
        pvec[:,0] = -np.sin(phi).ravel()  
        pvec[:,1] =  np.cos(phi).ravel()
        pvec[:,2] = 0 
        pvec[:,3] = 0.

        xyzw = rvec*radius 

        # check orthogonality of the frames at every point 
        check_pvec_tvec = np.abs(np.sum( pvec*tvec, axis=1 ))  
        check_pvec_rvec = np.abs(np.sum( pvec*rvec, axis=1 ))  
        check_tvec_rvec = np.abs(np.sum( tvec*rvec, axis=1 ))  

        assert check_pvec_tvec.max() < 1e-6 
        assert check_pvec_rvec.max() < 1e-6 
        assert check_tvec_rvec.max() < 1e-6 

        self.rtp = rtp
        self.xyzw = xyzw 
        self.rvec = rvec
        self.pvec = pvec
        self.tvec = tvec

    def __repr__(self):
        return "rtp %s rvec %s tvec %s pvec %s " % (str(self.rtp), str(self.rvec), str(self.tvec), str(self.pvec))

    def pvplot(self, pl):
        s = self
        pl.add_points( s.xyzw[:,:3], color=_white )
        pl.add_arrows( s.xyzw[:,:3], s.rvec[:,:3], color=_red )
        pl.add_arrows( s.xyzw[:,:3], s.tvec[:,:3], color=_green )
        pl.add_arrows( s.xyzw[:,:3], s.pvec[:,:3], color=_blue )


if __name__ == '__main__':

    sg = Spherical.Grid()
    s1 = Spherical.One(radius=22.)

    pl = pv.Plotter(window_size=SIZE*2 )

    sg.pvplot(pl)
    s1.pvplot(pl)

    cp = pl.show()



