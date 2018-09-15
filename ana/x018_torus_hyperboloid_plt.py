#!/usr/bin/env python
"""
Continuing from tboolean-12

Next:

1. ad-hoc overlay hyperboloid curves

2. consider ellipse tangent matching in addition to intersect : 
   looking at shape of the hyperboloid neck when targetting 
   closest approach point vs the torus neck : there seems to 
   be no point in fiddling more : also probably not possible
   as no other parameter to diddle 

3. Hype following the pattern of Ellipsoid etc.. hmm not so easy no
   nice patch ready to use : would have to approximate with a parametrically 
   defined path : so dont bother

4. Come up with G4Hype arguments and test how it looks in 3D, by changing x019.cc
  
::

     79 
     80   G4Hype(const G4String& pName,
     81                G4double  newInnerRadius,
     82                G4double  newOuterRadius,
     83                G4double  newInnerStereo,
     84                G4double  newOuterStereo,
     85                G4double  newHalfLenZ);
     86 

    // arghh : G4Hype cannot be z-cut : so would have to intersect with cylinder 
    // this makes me want to use a cone for the neck instead 


"""

import logging
log = logging.getLogger(__name__)
import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, PathPatch
import matplotlib.lines as mlines
import matplotlib.path as mpath

from opticks.ana.torus_hyperboloid import Tor, Hyp


class Shape(object):
    """
    matplotlib patches do not support deferred placement it seems, 
    so do that here 
    """
    KWA = dict(fill=False)
    dtype = np.float64

    PRIMITIVE = ["Ellipsoid","Tubs","Torus", "Cons", "Hype"]
    COMPOSITE = ["UnionSolid", "SubtractionSolid", "IntersectionSolid"]

    def __repr__(self):
        return "%s : %20s : %s : %s " % (
                      self.name, 
                      self.shape, 
                      repr(self.ltransform),
                      repr(self.param)
                     )


    def __init__(self, name, param, **kwa ):
        shape = self.__class__.__name__
        assert shape in self.PRIMITIVE + self.COMPOSITE
        primitive = shape in self.PRIMITIVE
        composite = shape in self.COMPOSITE

        d = self.KWA.copy()
        d.update(kwa)
        self.kwa = d

        self.name = name
        self.shape = shape      
        self.param = param

        self.parent = None
        self.ltransform = None
        self.left = None
        self.right = None

        if composite:  
            left = self.param[0]
            right = self.param[1]
            right.ltransform = self.param[2]  

            left.parent = self
            right.parent = self

            self.left = left
            self.right = right
        pass


    is_composite = property(lambda self:self.left is not None and self.right is not None)

    def _get_xy(self):
        xy = np.array([0,0], dtype=self.dtype )
        node = self
        while node is not None:
            if node.ltransform is not None:
                print "adding ltransform ", node
                xy += node.ltransform
            pass
            node = node.parent 
        pass
        return xy 

    xy = property(_get_xy)

    def patches(self):
        if self.shape == "Ellipsoid":
            return self.make_ellipse( self.xy, self.param, **self.kwa )
        elif self.shape == "Tubs":
            return self.make_rect( self.xy, self.param, **self.kwa)
        elif self.shape == "Torus":
            return self.make_torus( self.xy, self.param, **self.kwa)
        elif self.shape == "Cons":
            return self.make_cons( self.xy, self.param, **self.kwa)
        elif self.shape == "Hype":
            return self.make_hype( self.xy, self.param, **self.kwa)
        else:
            assert self.is_composite 
            pts = []
            pts.extend( self.left.patches() )
            pts.extend( self.right.patches() )
            return pts 
        pass

    @classmethod
    def make_rect(cls, xy , wh, **kwa ):
        """
        :param xy: center of rectangle
        :param wh: halfwidth, halfheight
        """
        ll = ( xy[0] - wh[0], xy[1] - wh[1] )
        return [Rectangle( ll,  2.*wh[0], 2.*wh[1], **kwa  )]

    @classmethod
    def make_ellipse(cls, xy , param, **kwa ):
        return [Ellipse( xy,  width=2.*param[0], height=2.*param[1], **kwa  )]

    @classmethod
    def make_circle(cls, xy , radius, **kwa ):
        return [Circle( xy,  radius=radius, **kwa  )] 

    @classmethod
    def make_torus(cls, xy, param, **kwa ):
        r = param[0]
        R = param[1]

        pts = []
        lhs = cls.make_circle( xy + [-R,0], r, **kwa)
        rhs = cls.make_circle( xy + [+R,0], r, **kwa)
        pts.extend(lhs)
        pts.extend(rhs)
        return pts

    @classmethod
    def make_pathpatch(cls, xy, vtxs, **kwa ):
        """see analytic/pathpatch.py"""
        Path = mpath.Path
        path_data = []
        
        for i, vtx in enumerate(vtxs):
            act = Path.MOVETO if i == 0 else Path.LINETO
            path_data.append( (act, (vtx[0], vtx[1])) )
        pass
        path_data.append( (Path.CLOSEPOLY, (vtxs[0,0], vtxs[0,1])) )
        pass

        codes, verts = zip(*path_data)
        path = Path(verts, codes)
        patch = PathPatch(path, **kwa)  
        return [patch]

    @classmethod
    def make_cons(cls, xy , param, **kwa ):
        r1 = param[0]
        r2 = param[1]
        hz = param[2]

        z2 =  hz + xy[1] 
        z1 = -hz + xy[1]

        vtxs = np.zeros( (4,2) )
        vtxs[0] = ( -r1, z1) 
        vtxs[1] = ( -r2,  z2)
        vtxs[2] = (  r2,  z2)
        vtxs[3] = (  r1,  z1)
        return cls.make_pathpatch( xy, vtxs, **kwa )


    @classmethod
    def make_hype(cls, xy , param, **kwa ):
        """
             4----------- 5
              3          6
               2        7 
              1          8
             0 ---------- 9

           sqrt(x^2+y^2) =   r0 * np.sqrt(  (z/zf)^2  +  1 )

        """
        r0 = param[0]
        stereo = param[1]
        hz = param[2]
        zf = r0/np.tan(stereo)

        r_ = lambda z:r0*np.sqrt( np.square(z/zf) + 1. )

        nz = 20 
        zlhs = np.linspace( -hz, hz, nz )
        zrhs = np.linspace(  hz, -hz, nz )

        vtxs = np.zeros( (nz*2,2) )

        vtxs[:nz,0] = -r_(zlhs) + xy[0]
        vtxs[:nz,1] = zlhs + xy[1]

        vtxs[nz:,0] = r_(zrhs) + xy[0]
        vtxs[nz:,1] = zrhs + xy[1]

        return cls.make_pathpatch( xy, vtxs, **kwa )


class Ellipsoid(Shape):pass
class Tubs(Shape):pass
class Torus(Shape):pass
class Cons(Shape):pass
class Hype(Shape):pass
class UnionSolid(Shape):pass
class SubtractionSolid(Shape):pass
class IntersectionSolid(Shape):pass



class X018(object):
    """
    G4VSolid* make_solid()
    { 
        G4ThreeVector A(0.000000,0.000000,-23.772510);
        G4ThreeVector B(0.000000,0.000000,-195.227490);
        G4ThreeVector C(0.000000,0.000000,-276.500000);
        G4ThreeVector D(0.000000,0.000000,92.000000);


        G4VSolid* d = new G4Ellipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
        G4VSolid* g = new G4Tubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510, 0.000000, CLHEP::twopi) ; // 4
        G4VSolid* i = new G4Torus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000, -0.000175, CLHEP::twopi) ; // 4
        
        G4VSolid* f = new G4SubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
        G4VSolid* c = new G4UnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2

        G4VSolid* k = new G4Tubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000, 0.000000, CLHEP::twopi) ; // 2
        
        G4VSolid* b = new G4UnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
        G4VSolid* m = new G4Tubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4VSolid* a = new G4IntersectionSolid("PMT_20inch_inner1_solid0x4cb3610", b, m, NULL, D) ; // 0
        return a ; 
    } 


                           a

                      b             m(D)
                 
                c          k(C)

            d     f(B) 

                g(B)  i(B+A)

    """
    def __init__(self, mode=0):
        A = np.array( [0, -23.772510] )
        C = np.array( [0, -276.500000] )
        D = np.array( [0, 92.000000] )

        B0 = np.array( [0, -195.227490] )
        z0 = B0[1] + A[1]    ## offset of torus (ellipsoid frame)

        self.z0 = z0

        i = Torus("i", [ 52.010000, 97.000000] )   
        # 97-51.01 = 44.99 ( slightly inside the tube radius 45.01 ? accident or CSG planning ?)
        r = i.param[0] 
        R = i.param[1]

        self.r = r
        self.R = R

        d = Ellipsoid("d", [249.000, 179.000 ] )
        ex = d.param[0]
        ez = d.param[1]

        self.ex = ex
        self.ez = ez

        k = Tubs("k", [45.010000, 57.510000] )
        bx = k.param[0]
        self.bx = bx


        p = ellipse_closest_approach_to_point( ex, ez, [R,z0] ) # center of torus RHS circle 
        pr = p[0]
        pz = p[1]    # z of ellipse to torus-circle closest approach point (ellipsoid frame) 

        self.p = p 


        if mode == 0:
            g = Tubs("g", [75.951247,23.782510] )
            f = SubtractionSolid("f", [g,i,A] )
            B = B0
        elif mode == 1:
            r2 = 83.9935              # at torus/ellipse closest approach point 
            r1 = R - r               

            mz = (z0 + pz)/2.         # mid-z cone coordinate (ellipsoid frame) 
            hz = (pz - z0)/2.         # cons half height  

            f = Cons("f", [r1,r2,hz ] )
            B = np.array( [0, mz] )

        elif mode == 2:

            r0 = R - r
            rw,zw = pr,pz-z0   # target the closest approach point  

            zf = Hyp.ZF( r0, zw, rw )
            hyp = Hyp( r0, zf )
         
            stereo = hyp.stereo
            hhh = pz - z0    # hype-half-height 

            f = Hype("f", [r0,stereo,hhh] ) 
            B = np.array( [0, z0] )

        else:
            assert 0 
        pass

        c = UnionSolid("c", [d,f,B] )

        m = Tubs("m", [254.000000, 92.000000] )
        b = UnionSolid("b", [c,k,C] )
        a = IntersectionSolid("a", [b,m,D] )

        self.root = a 
        self.sz = 400

        self.ellipse = d
        self.torus = i
        #self.cyneck = g
        self.f = f 
    pass


def ellipse_closest_approach_to_point( ex, ez, _c ):
    """ 
    Ellipse natural frame

    :param ex: semi-major axis 
    :param ez: semi-major axis 
    :param c: xz coordinates of point 
    """

    # parametric ellipse
    t = np.linspace( 0, 2*np.pi, 1000000 )
    e = np.zeros( [len(t), 2] )
    e[:,0] = ex*np.cos(t) 
    e[:,1] = ez*np.sin(t) 

    c = np.asarray( _c )   # center of RHS torus circle

    # point on ellipse of closest approach to center of torus circle
    p = e[np.sum(np.square(e-c), 1).argmin()]   

    return p 


if __name__ == '__main__':

    #x = X018(mode=0)  # crazy cylinder-torus neck
    #x = X018(mode=1)  # cone neck 
    x = X018(mode=2)  # hype neck 

    print "x.f ", x.f


    sz = x.sz

    r = x.r
    R = x.R
    z0 = x.z0   # natural torus frame offset in ellipse frame

    #ch = x.cyneck.param[1]*2   # full-height of cylinder neck 
    #cr = x.cyneck.param[0]     # half-width of cylinder neck, ie the radius 

    ex = x.ex
    ez = x.ez

    p = x.p  # closest approach of RHS torus circle center to a point on the ellipse  


    tor = Tor(R,r)
    assert tor.rz(0) == R - r 
    assert tor.rz(r) == R  
    assert np.all(tor.rz([0,r]) == np.asarray( [R-r, R] ) )


    tz0 = 0 
    tz1 = r      # how far in z to draw the torus and hype parametric lines
    #tz1 = ch 

    tz = np.linspace( tz0, tz1, 100)
    tr = tor.rz(tz)


    r0 = R - r     # Hyperboloid waist radius at z=0 (natural frame)
    # defining parameter of Hyperboloid by targetting an rz-point (natural frame)
    
    #rw,zw = R, r   # target the top of the torus : gives wider neck than  cy-tor 
    rw,zw = p[0],p[1]-z0   # target the closest approach point  
    halfZlen = p[1]-z0

    """
    hmm the intersection point of the torus "circle" and the ellipse would
    be a good point to target : as aiming to elimnate the cyclinder neck anyhow

    equation of RHS torus circle, in ellipse frame

        (x - R)^2 + (z - z0)^2 - r^2 = 0  

    equation of ellipse

        (x/ex)^2 + (z/ez)^2 - 1 = 0 

    """

    zf = Hyp.ZF( r0, zw, rw )
    hyp = Hyp( r0, zf )
    print hyp
    print "hyp halfZLen ", halfZlen 

    hr = hyp.rz(tz)

    tz += z0    


    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("x018_torus_hyperboloid_plt")

    ax = fig.add_subplot(111)
    ax.set_ylim([-350,200])
    ax.set_xlim([-300,300])

    for pt in x.root.patches():
        print "pt ", pt
        ax.add_patch(pt)
    pass

    #rhs = Circle( [R,z0], radius=r )
    #ax.add_patch(rhs)
    #ell = Ellipse( [0,0], 2*ex, 2*ez ) 
    #ax.add_patch(ell)

    #ax.scatter( e[:,0], e[:,1], s=1 ) 

    ax.scatter( p[0], p[1] , marker="*")
    ax.scatter( -p[0], p[1] , marker="*" )

    ax.plot( [R, p[0]], [z0, p[1]] )
    ax.plot( [-R, -p[0]], [z0, p[1]] )

    ax.plot( [R-r, p[0]], [z0, p[1]] )
    ax.plot( [-R+r, -p[0]], [z0, p[1]] )

    ax.plot(  tr, tz )
    ax.plot( -tr, tz )

    ax.plot(  hr, tz , linestyle="dashed")
    ax.plot( -hr, tz , linestyle="dashed")



   
    fig.show()


