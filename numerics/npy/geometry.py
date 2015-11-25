#!/usr/bin/env python

import os, logging
import numpy as np
import env.numerics.npy.PropLib as PropLib 
from StringIO import StringIO

log = logging.getLogger(__name__)


def mat4_tostring(m):
    from StringIO import StringIO
    sio = StringIO()
    np.savetxt(sio,m,delimiter=',',fmt="%.3f")
    s = sio.getvalue().replace("\n",",")
    return s 

def mat4_fromstring(s):
    m = np.fromstring(s,sep=",").reshape(4,4)
    return m 


class Material(object):
    def __init__(self, name):
        self.name = name
        self.mlib = None

    def refractive_index(self, wavelength):
        if self.mlib is None:
            self.mlib = PropLib.PropLib("GMaterialLib")
        pass
        n = self.mlib.interp(self.name,wavelength,PropLib.M_REFRACTIVE_INDEX)
        return n 

class Boundary(object):
    def __init__(self, spec):
         elem = spec.split("/")
         assert len(elem) == 4
         omat, osur, isur, imat = elem

         self.omat = Material(omat)
         self.osur = osur
         self.isur = isur
         self.imat = Material(imat)
         self.spec = spec


    def __repr__(self):
         return "%s %s " % (  self.__class__.__name__ , self.spec )

class Shape(object):
    def __init__(self, parameters, boundary):
         self.parameters = np.fromstring(parameters, sep=",")
         self.boundary = Boundary(boundary)

    def refractive_index(self, wavelength):
         return self.boundary.imat.refractive_index(wavelength)

    def __repr__(self):
         return "%s %s %s " % ( self.__class__.__name__ , self.parameters, self.boundary ) 


class Ray(object):
    def __init__(self, o, d):
        self.origin = np.array(o) 
        self.direction = np.array(d) 

    def position(self, t):
        return self.origin + t*self.direction


class Plane(object):
    def __init__(self, n, p):
        """
        :param n: normal 
        :param p: point in the plane 
        """
        n = np.array(n)
        p = np.array(p)
        n = n/np.linalg.norm(n,2,0)
        d = -np.dot(n,p)
        pass
        self.n = n
        self.p = p
        self.d = d

    def ntile(self, num):
        return np.tile( self.n, num).reshape(-1,3)

    def intersect(self, ray):
        denom_ = np.dot( self.n, ray.direction )
        return denom_, -(self.d + np.dot(self.n, ray.origin))/denom_







class Intersect(object):
    def __init__(self, i, t, n, p):
        self.i = i
        self.t = t 
        self.n = n
        self.p = p

    def __repr__(self):
        return "i %2d t %10.4f n %25s p %25s  " % (self.i, self.t, self.n,self.p)
 



class IntersectFrame(object):
    """
    world_to_intersect 4x4 homogenous matrix
    intersect frame basis 
    
    Arrange frame with origin at 
    intersection point and with basis vectors

    U: in plane of face
    V: normal to the intersected plane
    W: perpendicular to both the above

                 V
                 |
                 |
          -------*---U------A-
                W           another      
    """
    def __init__(self, isect, a):

        p = isect.p
        n = isect.n

        norm_ = lambda v:v/np.linalg.norm(v)  

        U = norm_( a - p )
        V = n
        W = np.cross(U,V)

        ro = np.identity(4)
        ro[:3,0] = U
        ro[:3,1] = V
        ro[:3,2] = W

        tr = np.identity(4)  
        tr[:3,3] = p

        itr = np.identity(4)  
        itr[:3,3] = -p

        self.w2i = np.dot(ro.T,itr)
        self.i2w = np.dot(tr, ro)  

        self.p = p
        self.n = n 
        self.a = a



    def homogenize(self, v, w=1):
        assert len(v) == 3
        vh = np.zeros(4)
        vh[:3] = v 
        vh[3] = w
        return vh

    def world_to_intersect(self, v, w=1): 
        wfp = self.homogenize(v, w)
        ifp = np.dot( self.w2i, wfp )
        log.info("world_to_intersect  wfp %s -> ifp %s " % (wfp, ifp)) 
        return ifp 

    def intersect_to_world(self, v, w=1): 
        ifp = self.homogenize(v, w)
        wfp = np.dot( self.i2w, ifp )
        log.info("intersect_to_world  ifp %s -> wfp %s " % (ifp, wfp)) 
        return wfp
 
    def i2w_string(self):
        return mat4_tostring(self.i2w.T)

    def id_string(self):
        return mat4_tostring(np.identity(4))







if __name__ == '__main__':
    pass
