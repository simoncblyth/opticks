#!/usr/bin/env python
"""

https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere

"""
import numpy as np


def xy_grid_coordinates(nx=11, ny=11, sx=100., sy=100.):
    """ 
    :param nx:
    :param ny:
    :param sx:
    :param sy:
    :return xyz: (nx*ny,3) array of XY grid coordinates
    """
    x = np.linspace(-sx,sx,nx)
    y = np.linspace(-sy,sy,ny)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros( (nx, ny) )
    xyz = np.dstack( (xx,yy,zz) ).reshape(-1,3)  
    return xyz 



def sample_linear(n, lo=-1., hi=1.):
    """
    :param n: number of values
    :param lo: min value
    :param hi: max value
    :return ll: array of shape (n,) with random sample values uniformly between lo and hi
    """
    mi = (lo + hi)/2.
    ex = (hi - lo)/2.
    uu = 2.*np.random.rand(n) - 1.   # -1->1 
    ll = mi + uu*ex      
    return ll 

def sample_linspace(n, lo=-1., hi=1. ):
    return np.linspace(lo, hi, n ) 


def sample_disc(n, dtype=np.float64):
    """
    https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly

    NB the sqrt(u) for the radius avoids clumping in the middle

    Circumference of circle 2.pi.r, area pi.r^2


    PDX(x) = 2x       Density for uniform radius sampling 

                 + 2
                /
               /
              /    
             / 
            /    + 1  
           /    
          /      
         /     
        +--------+ 
        0        1


    CDF(x) = Integral 2x dx = x^2     
    ICDF   sqrt(x)    


    """
    a = np.zeros( (n, 3), dtype=dtype ) 
    r = np.sqrt(np.random.rand(n))  
    t = 2.*np.pi*np.random.rand(n)
    a[:,0] = r*np.cos(t)
    a[:,1] = r*np.sin(t)
    return a 
    



 


def sample_trig(n):
    """
    :param n: number of points 
    """
    theta = 2*np.pi*np.random.rand(n)
    phi = np.arccos(2*np.random.rand(n)-1)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x,y,z])

def sample_normals(npoints):
    vec = np.random.randn(3, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_reject(npoints):
    vec = np.zeros((3,npoints))
    abc = 2*np.random.rand(3,npoints)-1
    norms = np.linalg.norm(abc,axis=0) 
    mymask = norms<=1
    abc = abc[:,mymask]/norms[mymask]
    k = abc.shape[1]
    vec[:,0:k] = abc
    while k<npoints:
       abc = 2*np.random.rand(3)-1
       norm = np.linalg.norm(abc)
       if 1e-5 <= norm <= 1:  
           vec[:,k] = abc/norm
           k = k+1
    return vec


if __name__ == '__main__':

    n = 10 
    a = sample_trig(n)
    b = sample_normals(n)
    c = sample_reject(n)


