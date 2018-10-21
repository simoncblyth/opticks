#!/usr/bin/env python
"""
TODO: get this to work with python3
"""
import logging

import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, PathPatch
import matplotlib.lines as mlines
import matplotlib.path as mpath


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
                print("adding ltransform ", node)
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



if __name__ == '__main__':
    pass







