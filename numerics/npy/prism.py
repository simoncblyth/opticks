#!/usr/bin/env python
"""
Hecht p163, two refractions thru a prism, CB and CD are normal to surface planes::

   .                
               
                    A  
                   / \
                  /   \
                 /     \
                /       \
               /         \    
              /           \   
             /             \
            B. . . . . . . .D   
       .   /                 \
          /          C        \
         /                     \ 
        /                       \
       --------------------------


Polygon ABCD

   BAD : alpha   (apex angle)
   CBA : 90 degrees
   CDA : 90 degrees
   BCD : 180 - alpha 

Triangle BCD has angles:

   CBD: t1 
   CDB: i2
   BCD: 180 - alpha   

  ==>  alpha = t1 + i2    


Ray path thru prism, BD

Deviation at B:   (i1 - t1) 
Deviation at D:   (t2 - i2)
Total    
         delta:   (i1 - t1) + (t2 - i2)

         delta = i1 + t2 - alpha     

Aiming for expression providing delta as function of theta_i1, 
apex angle alpha and refractive index n 

Snells law at 2nd interface, prism refractive index n in air  

   sin(t2) = n sin(i2) 

       t2  = arcsin( n sin(i2) )
           = arcsin( n sin(alpha - t1) )
           = arcsin( sin(alpha) n cos(t1) - cos(alpha) n sin(t1) )


      [ n cos(t1) ]^2  = (n^2 - n^2 sin(t1)^2 )

                       =  (n^2 - sin(i1)^2 )           n sin(t1) = sin(i1)


       t2 = arcsin(  sin(alpha) sqrt(n^2 - sin^2(i1)) - sin(i1) cos(alpha) )
     

2nd refraction has a critical angle where t2 = pi above which TIR will occur

       n sin(i2) = 1*sin(t2) =  1  

             i2c = arcsin( 1./n )


Propagate that back to 1st refraction

         sin(i1) = n sin(t1) = n sin(alpha - i2)

             i1  = arcsin( n sin(alpha - arcsin(1/n) ) ) 
 

But there is another constraint in there with edge 

             n sin(alpha - arcsin(1/n)) = 1
                   alpha - arcsin(1/n) = arcsin(1/n)

                           alpha/2 = arcsin(1/n) = i2c

                           alpha = 2*i2c      (i2_c  43.304 n = 1.458)

This indicates that a 90 degree apex angle is not a good choice 
for dispersing prism, use 60 degree instead.



At minimum deviation delta, ray are parallel to base and have symmetric ray 

    i1 = t2 
    t1 = i2

    alpha = t1 + i2         ==>  t1 = i2 = alpha/2
   
    delta = i1 + t2 - alpha     


    sin(delta + alpha) = sin( i1 + t2 ) 

    sin(i1) = n sin(t1)  

        i1 = arcsin( n sin(alpha/2) )


Where to shoot from to get minimum deviation ?

      



"""

import os, logging
import numpy as np
rad = np.pi/180.
from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

from env.numerics.npy.ana import Evt, Selection, Rat, theta
from env.numerics.npy.fresnel import Fresnel


np.set_printoptions(suppress=True, precision=3)
np.seterr(divide="ignore", invalid="ignore")
# http://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.seterr.html

rat_ = lambda n,d:float(len(n))/float(len(d))

X,Y,Z,W = 0,1,2,3





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
    def __init__(self, i0, t0, n0 ,p0, i1, t1, n1, p1):
        self.i0 = i0
        self.t0 = t0 
        self.n0 = n0
        self.p0 = p0

        self.i1 = i1
        self.t1 = t1
        self.n1 = n1 
        self.p1 = p1

    def __repr__(self):
        return "i0 %2d t0 %10.4f n0 %25s p0 %25s   i1 %2d t1 %10.4f n1 %25s p1 %25s " % (self.i0, self.t0, self.n0,self.p0, self.i1, self.t1, self.n1, self.p1 )
 


class Prism(object):
    def __init__(self, alpha, n, height=300., depth=300.):
        a  = alpha*rad
        pass
        self.a = a 
        self.sa = np.sin(a)
        self.ca = np.cos(a)
        self.n = n
        self.nn = n*n
        self.ni = 1./n
        self.alpha = alpha


        ymax =  height/2.
        ymin = -height/2.

        hwidth = height*np.tan(a/2.) 
        apex = [0,ymax,0]
        base = [0,ymin,0]
        front = [0,ymin,depth/2.]
        back  = [0,ymin,-depth/2.]

        lhs = Plane([-height, hwidth, 0], apex )
        rhs = Plane([ height, hwidth, 0], apex ) 
        bot = Plane([      0,     -1, 0], base )
        bck = Plane([      0,      0, -1], back )
        frt = Plane([      0,      0,  1], front )
 
        self.lhs = lhs
        self.rhs = rhs
        self.bot = bot
        self.ymin = ymin

        self.planes = [rhs, lhs, bot, frt, bck ]
        self.height = height
        self.depth = depth
        self.apex = np.array(apex)


    def intersect(self, ray):
         t0 = -np.inf
         t1 =  np.inf
         t0_normal = np.array([0,0,0])
         t1_normal = np.array([0,0,0])

         for i, pl in enumerate(self.planes):
             n = pl.n

             denom, t = pl.intersect(ray)

             if denom < 0.:
                if t > t0:
                    i0 = i 
                    t0 = t 
                    t0_normal = n  
             else:
                 if t < t1:
                    i1 = i 
                    t1 = t 
                    t1_normal = n 

             log.info("i %2d denom %10.4f t %10.4f t0 %10.4f t1 %10.4f t0n %25s t1n %25s " % (i, denom, t, t0, t1, t0_normal, t1_normal ))

         if t0 > t1:
             log.into("no intersect")
             return None
         else:
             p0 = ray.position(t0)
             p1 = ray.position(t1)
             return Intersect(i0, t0, t0_normal, p0,  i1, t1, t1_normal, p1)



    def intersect_unrolled(self, ray):
        """
        http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/

        # 0 * inf = nan   comparisons with nan always false

        """
        pl = self.planes[0]
        t0 = pl

    def expected(self):
        _i1c = self.i1c()/rad
        _i2c = self.i2c()/rad
        print "_i1c ", _i1c
        print "_i2c ", _i2c
        dom = self.i1_domain()
        dl = self.delta(dom*rad)/rad
        print "plt.plot(dom,dl);"
        return dom, dl

    def i1_domain(self, num=100):
        _i1c = self.i1c()/rad
        return np.linspace(_i1c+1e-9,90,200)
 
    def delta(self, i1):
        return i1 + self.t2(i1) - self.a

    def i1m(self):
        """
        incident angle with minimum deviation
        """ 
        return np.arcsin( self.n*np.sin(self.a/2.) )

    def i2c(self):
        return np.arcsin(1./self.n)

    def i1c(self):
        return np.arcsin( self.n*np.sin(self.a - np.arcsin(self.ni)))

    def sint2(self, i1):
        return self.sa*np.sqrt(self.nn - np.sin(i1)*np.sin(i1)) - np.sin(i1)*self.ca  

    def t2(self, i1):
        return np.arcsin(self.sa*np.sqrt(self.nn - np.sin(i1)*np.sin(i1)) - np.sin(i1)*self.ca)  

       

dot_ = lambda a,b:np.sum(a * b, axis = 1)/(np.linalg.norm(a, 2, 1)*np.linalg.norm(b, 2, 1)) 
norm_ = lambda v:v/np.linalg.norm(v)  

def hom_(v):
    h = np.ones(4)
    h[:3] = v 
    return h 


class PrismCheck(object):
    def __init__(self, prism, evt=None ):

        if evt is None:
            evt = Evt(tag="1")
      
        s = Selection(evt,"BT BT SA")   # smth like 70 percent 

        p0 = s.recpost(0)[:,:3]  # light source position  
        p1 = s.recpost(1)[:,:3]  # 1st refraction point
        p2 = s.recpost(2)[:,:3]  # 2nd refraction point
        p3 = s.recpost(3)[:,:3]  # light absorption point
     
        assert len(p0) == len(p1) == len(p2) == len(p3)
        N = len(p0)

        p01 = p1 - p0
        p12 = p2 - p1 
        p23 = p3 - p2

        cdv = dot_( p01, p23 )    # total deviation angle
        dv = np.arccos(cdv)

        # below assumes a particular intersection 
                 
        lno = prism.lhs.ntile(N)  # prism lhs normal, repeated
        rno = prism.rhs.ntile(N)  # prism rhs normal, repeated

        ci1 = dot_(-p01, lno )    # incident 1 
        ct1 = dot_(-p12, lno )    # transmit 1 
        ci2 = dot_( p12, rno )    # incident 2
        ct2 = dot_( p23, rno )    # transmit 2

        i1 = np.arccos(ci1)
        t1 = np.arccos(ct1)
        i2 = np.arccos(ci2)
        t2 = np.arccos(ct2)

        # Snell check 
        n1 = np.sin(i1)/np.sin(t1)
        n2 = np.sin(t2)/np.sin(i2)

        
        ddv = prism.delta(i1) - dv    # compare expected deviation with simulation result

        self.prism = prism 
        self.evt = evt 



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
    def __init__(self, isect, another):

        p0 = isect.p0
        n0 = isect.n0

        U = norm_( another - p0 )
        V = n0
        W = np.cross(U,V)

        ro = np.identity(4)
        ro[:3,0] = U
        ro[:3,1] = V
        ro[:3,2] = W

        tr = np.identity(4)  
        tr[:3,3] = p0

        itr = np.identity(4)  
        itr[:3,3] = -p0

        self.w2i = np.dot(ro.T,itr)
        self.i2w = np.dot(tr, ro)  


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
 



if __name__ == '__main__':

    prism = Prism(60., 1.458)
    dom, dl = prism.expected()

    i1m = prism.i1m()
    ci1m = np.cos(i1m)
    si1m = np.sin(i1m)
    ti1m = np.tan(i1m)
    print "i1m %s ci1m %s " % (i1m, ci1m)

    # light source and target basis, to which random disc position is added     
    src = np.array([[-600,0,0]])
    tgt = np.array([[0,0,0]])

    lnorm = prism.lhs.ntile(1)   # particular intersection assumption
    cta = dot_( src-tgt, lnorm )

    
    middle = Ray( [-600,0,0], [1,0,0] )
    graze = Ray( [-600, prism.ymin, 0], [1,0,0])

    ray = middle

    isect = prism.intersect(ray)   
    if isect is None:
        log.info("no intersect")
    else:
        log.info(isect)
        log.info("p0 %s n0 %s " % (isect.p0,isect.n0))

        another = prism.apex  
        # another point in the plane of the intersected face, 
        # in order to define intersect face basis
        ifr = IntersectFrame( isect, another)

        ifr.world_to_intersect(isect.p0)      
        ifr.world_to_intersect(isect.n0, w=0)     # direction, not coordinate 
        ifr.world_to_intersect([0,0,1], w=0)     # direction, not coordinate 
 
        ifr.intersect_to_world([0,0,0])

        ifpm = np.array([-1, 1./ti1m,0])*400
        ifr.intersect_to_world(ifpm)
        







    




