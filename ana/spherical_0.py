#!/usr/bin/env python 
"""
spherical_0.py
================

See spherical.py and tangential.py for further development on this theme.

https://mathworld.wolfram.com/SphericalCoordinates.html

Draws points on a sphere together with a (r,phi,theta) derivative vectors
which are normals and tangents to the sphere and form orthogonal 
frames at every point on the sphere.

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

"""
import numpy as np
import pyvista as pv
SIZE = np.array([1280, 720])
DTYPE = np.float64

if __name__ == '__main__':

     radius = 20.
     v_phi = np.pi*np.linspace(0,2,50, dtype=DTYPE) 
     v_theta = np.pi*np.linspace(0,1,50, dtype=DTYPE )   

     num = len(v_theta)*len(v_phi) 

     theta, phi = np.meshgrid(v_theta, v_phi)
     print("theta.shape %s " % str(theta.shape))
     print("phi.shape %s " % str(phi.shape))

     rvec = np.zeros( (num,3),  dtype=DTYPE )
     pvec = np.zeros( (num,3),  dtype=DTYPE )
     tvec = np.zeros( (num,3),  dtype=DTYPE )

     # derivative with radius of definition of spherical coordinates : direction of increasing radius : normal vector
     x = np.cos(phi)*np.sin(theta)
     y = np.sin(phi)*np.sin(theta)
     z = np.cos(theta)

     rvec[:,0] = x.ravel()
     rvec[:,1] = y.ravel() 
     rvec[:,2] = z.ravel() 
     pos = rvec*radius 

     r = np.sqrt(x*x + y*y + z*z)

     # derivative with phi : direction of increasing phi : phi tangent vector
     pvec[:,0] = -np.sin(phi).ravel()  
     pvec[:,1] =  np.cos(phi).ravel()
     pvec[:,2] = 0 

     # derivative with theta  : direction of increasing theta : theta tangent vector
     tvec[:,0] =  (np.cos(phi)*np.cos(theta)).ravel()
     tvec[:,1] =  (np.sin(phi)*np.cos(theta)).ravel()
     tvec[:,2] = -np.sin(theta).ravel() 

     # check orthogonality of the frames at every point 
     check_pvec_tvec = np.abs(np.sum( pvec*tvec, axis=1 ))  
     check_pvec_rvec = np.abs(np.sum( pvec*rvec, axis=1 ))  
     check_tvec_rvec = np.abs(np.sum( tvec*rvec, axis=1 ))  

     assert check_pvec_tvec.max() < 1e-6 
     assert check_pvec_rvec.max() < 1e-6 
     assert check_tvec_rvec.max() < 1e-6 

     pl = pv.Plotter(window_size=SIZE*2 )

     _white = "ffffff"
     _red = "ff0000"
     _green = "00ff00"
     _blue = "0000ff"

     pl.add_points( pos,       color=_white )
     pl.add_arrows( pos, rvec, color=_red )
     pl.add_arrows( pos, tvec, color=_green )
     pl.add_arrows( pos, pvec, color=_blue )

     cp = pl.show()


