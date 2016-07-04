#!/usr/bin/env python
"""
TODO:

* handle multiple wavelengths
* make comparison after histogramming, like reflection.py does
* but here have incident angle as well as the deviation, so need 2d 
* http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.histogram2d.html


See *ggv-prism*


Prism Deviation Angle Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
* Use intersect frame coordinate with the transform explicitly specifified


"""

import os, logging
import numpy as np
rad = np.pi/180.
deg = 1./rad
from env.python.utils import *
from opticks.ana.types import *

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

from opticks.ana.ana import Evt, Selection, Rat, theta, costheta_
from opticks.ana.geometry import Shape, Plane, Boundary, Ray, Intersect, IntersectFrame, mat4_tostring, mat4_fromstring
import ciexyz.ciexyz as cie

deg = np.pi/180.


np.set_printoptions(suppress=True, precision=3)
np.seterr(divide="ignore", invalid="ignore")

rat_ = lambda n,d:float(len(n))/float(len(d))

X,Y,Z,W = 0,1,2,3


class Box(Shape):
    def __init__(self, parameters, boundary, wavelength=380 ):
        Shape.__init__(self, parameters, boundary) 


class Prism(Shape):
    def __init__(self, parameters, boundary):
        Shape.__init__(self, parameters, boundary) 

        alpha = self.parameters[0]
        height = self.parameters[1]
        depth = self.parameters[2]

        a  = alpha*rad
        pass
        self.a = a 
        self.sa = np.sin(a)
        self.ca = np.cos(a)

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


    def intersectframe(self, ray):
        """
        Form coordinate system based centered on intersection point, 
        with surface normal along Y. 
        Another point in the plane of the intersected face, 
        is used to pick the X direction.
        """
        isect = self.intersect(ray)   
        assert len(isect)>0
        i0, i1 = isect
        
        ifr = IntersectFrame( i0, a=prism.apex)

        ## check transforming from world frame into intersect frame
        p_if = ifr.world_to_intersect(ifr.p)      
        n_if = ifr.world_to_intersect(ifr.n, w=0)     # direction, not coordinate 
        a_if = ifr.world_to_intersect(ifr.a)      

        pa_wf = ifr.a - ifr.p
        pa_if = ifr.world_to_intersect(pa_wf, w=0)
        pa_if /= np.linalg.norm(pa_if) 
 
        log.info("intersect position       ifr.p %s in p_if  %s " % (ifr.p, p_if )) 
        log.info("intersect surface normal ifr.n %s in n_if  %s " % (ifr.n, n_if )) 
        log.info("                         ifr.a %s in a_if  %s " % (ifr.a, a_if )) 
        log.info("                         pa_wf %s in pa_if %s " % (pa_wf, pa_if )) 

        assert np.allclose( p_if, np.array([0,0,0,1]))
        assert np.allclose( n_if, np.array([0,1,0,0]))
        assert np.allclose( pa_if, np.array([1,0,0,0]))

        ## check transforming from intersect frame into world frame
        o_if = np.array([0,0,0])
        o_wf = ifr.intersect_to_world(o_if)
        log.info("                        o_if %s o_wf %s " % (o_if, o_wf )) 
        assert np.allclose( o_wf[:3], ifr.p )

        return ifr 



    def intersect(self, ray):
         t0 = -np.inf
         t1 =  np.inf
         n0 = np.array([0,0,0])
         n1 = np.array([0,0,0])

         for i, pl in enumerate(self.planes):
             n = pl.n

             denom, t = pl.intersect(ray)

             if denom < 0.:
                if t > t0:
                    i0 = i 
                    t0 = t 
                    n0 = n  
             else:
                 if t < t1:
                    i1 = i 
                    t1 = t 
                    n1 = n 

             log.info("i %2d denom %10.4f t %10.4f t0 %10.4f t1 %10.4f n0 %25s n1 %25s " % (i, denom, t, t0, t1, n0, n1 ))

         if t0 > t1:
             log.into("no intersect")
             return []
         else:
             p0 = ray.position(t0)
             p1 = ray.position(t1)
             return Intersect(i0, t0, n0, p0),Intersect(i1, t1, n1, p1)

    def intersect_unrolled(self, ray):
        """
        http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/

        # 0 * inf = nan   comparisons with nan always false

        """
        pl = self.planes[0]
        t0 = pl



class PrismExpected(object):
    def __init__(self, a, n):
        self.a = a
        self.sa = np.sin(a)
        self.ca = np.cos(a)

        self.single = type(n) is np.float64

        self.n = n
        self.nn = n*n
        self.ni = 1./n

        self.i2c = self.i2c_()
        self.i1c = self.i1c_()

    def i2c_(self):
        return np.arcsin(1./self.n)

    def i1c_(self):
        return np.arcsin( self.n*np.sin(self.a - np.arcsin(self.ni)))

    def st2_(self, i1):
        return self.sa*np.sqrt(self.nn - np.sin(i1)*np.sin(i1)) - np.sin(i1)*self.ca  

    def t2_(self, i1):
        return np.arcsin(self.sa*np.sqrt(self.nn - np.sin(i1)*np.sin(i1)) - np.sin(i1)*self.ca)  

    def delta_(self, i1):
        return i1 + self.t2_(i1) - self.a

    def i1mindev_(self):
        return np.arcsin( self.n*np.sin(self.a/2.) )  # incident angle with minimum deviation

    def i1_domain(self):
        assert self.single
        _i1c = self.i1c/rad
        _i2c = self.i2c/rad
        log.debug( "_i1c %s " % _i1c)
        log.debug( "_i2c %s " % _i2c)
        log.info("plt.plot(dom,dl)")
        return np.linspace(_i1c+1e-9,90,200)
    
    def expected(self):
        assert self.single
        dom = self.i1_domain()
        dl = self.delta_(dom*rad)/rad
        return dom, dl
 
    def spawn_singles(self):
        """
        For plotting need a domains and corresponding values, BUT must 
        split up by refractive index to avoid spagetti 
        """
        un = np.unique(self.n)
        if len(un) > 10:
           log.warn("too many distinct indices");
           return []

        return [PrismExpected(self.a, n) for n in un]



class PrismCheck(object):
    """
    Assumes canonical passage thru prism (ie not out the bottom) 
    """
    def title(self):
        return "prism.py deviation vs incident angle"  

    def __init__(self, prism, xprism, sel, mask=True):
      
        self.prism = prism 
        self.xprism = xprism 
        self.sel = sel

        p0 = sel.recpost(0)[:,:3]  # light source position  
        p1 = sel.recpost(1)[:,:3]  # 1st refraction point
        p2 = sel.recpost(2)[:,:3]  # 2nd refraction point
        p3 = sel.recpost(3)[:,:3]  # light absorption point
        assert len(p0) == len(p1) == len(p2) == len(p3) 
        N = len(p0)

        w0 = sel.recwavelength(0)
        w1 = sel.recwavelength(1)
        w2 = sel.recwavelength(2)
        w3 = sel.recwavelength(3)
        assert len(w0) == len(w1) == len(w2) == len(w3) 
        assert len(w0) == len(p0)
        assert np.all(w0 == w1)
        assert np.all(w0 == w2)
        assert np.all(w0 == w3)
        self.wx = w0
        # prism does spatial sorting, no wavelength changes as no reemission

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


        p01 = p1 - p0
        p12 = p2 - p1 
        p23 = p3 - p2

        self.p01 = p01
        self.p12 = p12
        self.p23 = p23

        cdv = costheta_( p01, p23 )    # total deviation angle
        dv = np.arccos(cdv)

        self.cdv = cdv 
        self.dv = dv    # simulated deviation

        lno = prism.lhs.ntile(N)  # prism lhs normal, repeated
        rno = prism.rhs.ntile(N)  # prism rhs normal, repeated

        ci1 = costheta_(-p01, lno )    # incident 1 
        ct1 = costheta_(-p12, lno )    # transmit 1 
        ci2 = costheta_( p12, rno )    # incident 2
        ct2 = costheta_( p23, rno )    # transmit 2

        i1 = np.arccos(ci1)
        t1 = np.arccos(ct1)
        i2 = np.arccos(ci2)
        t2 = np.arccos(ct2)

        self.i1 = i1
        self.t1 = t1
        self.i2 = i2
        self.t2 = t2

        # Snell check : refractive index from the angles at the 2 refractions
        n1 = np.sin(i1)/np.sin(t1)
        n2 = np.sin(t2)/np.sin(i2)
        dn = n1 - n2

        #assert dn.max() < 1e-3
        #assert dn.min() > -1e-3


        self.n1 = n1
        self.n2 = n2

        self.expected_deviation()
        #self.compare_expected_with_simulated()

    def expected_deviation(self):
        """  
        Get 8 nan close to critical angle

        In [108]: pc.i1[pc.inan]*deg
        Out[108]: 
        array([ 24.752,  24.752,  24.752,  24.752,  24.752,  24.752,  24.752,
                24.752])

        simulating just those around critical angle, 
        see that they are just sneaking out the prism grazing the face
        """ 
        i1 = self.i1

        xdv = self.xprism.delta_(i1)  
        ina = np.arange(len(xdv))[np.isnan(xdv)]
        msk = np.arange(len(xdv))[~np.isnan(xdv)]

        if len(ina)>0:
           log.warning("expected_deviation needs nan masking ina:%s len(ina):%s " % (ina, len(ina))) 

        self.xdv = xdv 
        self.ina = ina
        self.msk = msk 

        i1c = self.xprism.i1c_()

        log.info("ina: %s " % self.ina)
        log.info("i1[ina]*deg: %s " % (i1[self.ina]*deg) )
        log.info("i1c*deg : %s " % (i1c*deg) )

    def compare_expected_with_simulated(self):
        """
        """
        msk = self.msk
        mdf = self.dv[msk] - self.xdv[msk]
        log.info("  dv[msk]*deg %s " % (self.dv[msk]*deg) )
        log.info(" xdv[msk]*deg %s " % (self.xdv[msk]*deg) )
        log.info("mdf*deg:%s max:%s min:%s " % (mdf*deg, mdf.min()*deg, mdf.max()*deg ))

        self.mdf = mdf
        self.mdv = self.dv[msk]
        self.mi1 = self.i1[msk]




def test_intersectframe(prism):
    """
    Establish a frame at midface lhs intersection point with the prism 
    """ 
    ray = Ray([-600,0,0], [1,0,0]) 
    ifr = prism.intersectframe(ray)
    i1m = prism.i1mindev()
    ti1m = np.tan(i1m)
    pm_if =  np.array([-1, 1./ti1m,0])*400
    pm_wf =  ifr.intersect_to_world(pm_if)
    log.info(" mindev position pm_if %s pm_wf %s " % (pm_if, pm_wf ))
    s_i2w = ifr.i2w_string()
    log.info("  s_i2w %s " % s_i2w );  



def scatter_plot(xq, yq, sl):
    if sl is not None:
        x = xq[sl]*deg  
        y = yq[sl]*deg  
    else:
        x = xq*deg  
        y = yq*deg  
    pass
    plt.scatter(x, y)


def vanity_plot(pc, sl=None):
    for xprism in pc.xprism.spawn_singles():
        dom, dl = xprism.expected()
        plt.plot(dom,dl)

    scatter_plot(pc.i1, pc.dv, sl)

def deviation_plot(pc,sl=None):
    """
    masked plotting to avoid critical angle nan

    large (+-0.5 degree) deviations between expected and simulated delta 
    all occuring close to critical angle

    away from critical angle, deviations less that 0.1 degree
    """ 
    scatter_plot(pc.mi1, pc.mdf, sl)


def oneplot(pc, log_=False):
    fig = plt.figure()
    plt.title(pc.title())
    ax = fig.add_subplot(111)

    vanity_plot(pc, sl=slice(0,10000))
    #deviation_plot(pc, sl=slice(0,10000))
    #deviation_plot(pc, sl=None)

    fig.show()



def spatial(pc):
    """
    """
    w = pc.wx
    x = pc.p3[:,0]
    y = pc.p3[:,1]
    z = pc.p3[:,2]

    #assert np.all(x == 1200.)
    #off = x != 1200.   
    #print pc.p3[off] 

    from matplotlib.colors import LogNorm

    ax.hist2d( w, y, bins=100, norm=LogNorm())






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()

    #wl = np.arange(10, dtype=np.float32)*70. + 100.  
    # low range is off edge of the refractive index values 

    evt = Evt(tag="1", det="newton") # "prism" uses all incident angles, "newton" uses one

    sel = Selection(evt,"BT BT SA")  

    boundary = Boundary("Vacuum///GlassSchottF2")

    prism = Prism("60.,300,300,0", boundary)

    n = boundary.imat.refractive_index(sel.wl)  

    xprism = PrismExpected(prism.a, n)


    pc = PrismCheck(prism, xprism, sel )


    fig = plt.figure()
    ax = fig.add_subplot(111)



    oneplot(pc, log_=False)
    #spatial(pc)

  

    plt.show()



