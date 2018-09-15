#!/usr/bin/env python
"""
Continuing from tboolean-12
"""

import logging
log = logging.getLogger(__name__)
import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import matplotlib.lines as mlines


class Shape(object):
    """
    matplotlib patches do not support deferred placement it seems, 
    so do that here 
    """
    KWA = dict(fill=False)
    dtype = np.float64

    PRIMITIVE = ["Ellipsoid","Tubs","Torus"]
    COMPOSITE = ["UnionSolid", "SubtractionSolid", "IntersectionSolid"]

    def __repr__(self):
        return "%s : %s%s : %20s : %s " % (self.name, 
                                   ("L" if self.is_left_child else "l"),
                                   ("R" if self.is_right_child else "r"),
                                   self.shape, repr(self.ltransform) )


    def __init__(self, name, shape, param, **kwa ):
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
    is_right_child = property(lambda self:self.parent is not None and self.parent.right == self)
    is_left_child = property(lambda self:self.parent is not None and self.parent.left == self)

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
            return [self.make_ellipse( self.xy, self.param, **self.kwa )] 
        elif self.shape == "Tubs":
            return [self.make_rect( self.xy, self.param, **self.kwa)]
        elif self.shape == "Torus":
            r = self.param[0]
            R = self.param[1]
            lhs = self.make_circle( self.xy + [-R,0], r, **self.kwa)
            rhs = self.make_circle( self.xy + [+R,0], r, **self.kwa)
            return [lhs, rhs]
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
        return Rectangle( ll,  2.*wh[0], 2.*wh[1], **kwa  ) 
    @classmethod
    def make_ellipse(cls, xy , param, **kwa ):
        return Ellipse( xy,  width=2.*param[0], height=2.*param[1], **kwa  ) 
    @classmethod
    def make_circle(cls, xy , radius, **kwa ):
        return Circle( xy,  radius=radius, **kwa  ) 


class Ellipsoid(Shape):
    def __init__(self, name, param, **kwa):
        Shape.__init__(self, name, self.__class__.__name__, param, **kwa )
class Tubs(Shape):
    def __init__(self, name, param, **kwa):
        Shape.__init__(self, name, self.__class__.__name__, param, **kwa )
class Torus(Shape):
    def __init__(self, name, param, **kwa):
        Shape.__init__(self, name, self.__class__.__name__, param, **kwa )


class BooleanSolid(Shape):
    def __init__(self, name, param, **kwa):
        Shape.__init__(self, name, self.__class__.__name__, param, **kwa )


class UnionSolid(BooleanSolid):pass
class SubtractionSolid(BooleanSolid):pass
class IntersectionSolid(BooleanSolid):pass


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
    def __init__(self):
        A = np.array( [0, -23.772510] )
        B = np.array( [0, -195.227490] )
        C = np.array( [0, -276.500000] )
        D = np.array( [0, 92.000000] )

        g = Tubs("g", [75.951247,23.782510] )

        i = Torus("i", [ 52.010000, 97.000000] )
        f = SubtractionSolid("f", [g,i,A] )

        d = Ellipsoid("d", [249.000, 179.000 ] )
        c = UnionSolid("c", [d,f,B] )

        k = Tubs("k", [45.010000, 57.510000] )
        m = Tubs("m", [254.000000, 92.000000] )
        b = UnionSolid("b", [c,k,C] )
        a = IntersectionSolid("a", [b,m,D] )

        self.a = a 
        self.sz = 400
    pass




if __name__ == '__main__':

    x = X018()
    sz = x.sz

    plt.ion()
    fig = plt.figure(figsize=(5,5))
    plt.title("x018_torus_hyperboloid_plt")

    ax = fig.add_subplot(111)
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])

    for pt in x.a.patches():
        ax.add_patch(pt)
    pass
   
    fig.show()


