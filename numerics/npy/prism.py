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

rat_ = lambda n,d:float(len(n))/float(len(d))

X,Y,Z,W = 0,1,2,3



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



class Prism(object):
    def __init__(self, alpha, n, height=300.):
        a  = alpha*rad
        pass
        self.a = a 
        self.sa = np.sin(a)
        self.ca = np.cos(a)
        self.n = n
        self.nn = n*n
        self.ni = 1./n
        self.alpha = alpha

        hwidth = height*np.tan(a/2.) 
        apex = [0,height,0]

        self.lhs = Plane([-height, hwidth, 0], apex )
        self.rhs = Plane([ height, hwidth, 0], apex ) 
        self.height = height

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
  

if __name__ == '__main__':

    prism = Prism(60., 1.458)
    dom, dl = prism.expected()

    i1m = prism.i1m()
    ci1m = np.cos(i1m)
    print "i1m %s ci1m %s " % (i1m, ci1m)


    e = Evt(tag="1")
    s = Selection(e,"BT BT SA")   # smth like 70 percent 

    p0 = s.recpost(0)[:,:3]  # light source position  
    p1 = s.recpost(1)[:,:3]  # 1st refraction point
    p2 = s.recpost(2)[:,:3]  # 2nd refraction point
    p3 = s.recpost(3)[:,:3]  # light absorption point
 
    assert len(p0) == len(p1) == len(p2) == len(p3)
    N = len(p0)

    p01 = p1 - p0
    p12 = p2 - p1 
    p23 = p3 - p2
             
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

    cdv = dot_( p01, p23 )    # total deviation
    dv = np.arccos(cdv)

    # Snell check 
    n1 = np.sin(i1)/np.sin(t1)
    n2 = np.sin(t2)/np.sin(i2)

    
    ddv = prism.delta(i1) - dv    # compare expected deviation with simulation result


    # light source and target basis, to which random disc position is added     
    src = np.array([[-600,0,0]])
    tgt = np.array([[0,0,0]])
    cta = dot_( src-tgt, lno[0:1] )

    



    




