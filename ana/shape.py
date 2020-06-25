#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
TODO: get this to work with python3
"""
import logging, copy 

log = logging.getLogger(__name__)

import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, PathPatch
import matplotlib.lines as mlines
import matplotlib.path as mpath


def ellipse_closest_approach_to_point( ex, ez, _c ):
    """ 
    Ellipse natural frame, semi axes ex, ez.  _c coordinates of point

    :param ex: semi-major axis 
    :param ez: semi-major axis 
    :param c: xz coordinates of point 

    :return p: point on ellipse of closest approach to center of torus circle

    Closest approach on the bulb ellipse to the center of torus "circle" 
    is a good point to target for hype/cone/whatever neck, 
    as are aiming to eliminate the cylinder neck anyhow

    equation of RHS torus circle, in ellipse frame

        (x - R)^2 + (z - z0)^2 - r^2 = 0  

    equation of ellipse

        (x/ex)^2 + (z/ez)^2 - 1 = 0 

    """
    c = np.asarray( _c )   # center of RHS torus circle
    assert c.shape == (2,)

    t = np.linspace( 0, 2*np.pi, 1000000 )
    e = np.zeros( [len(t), 2] )
    e[:,0] = ex*np.cos(t) 
    e[:,1] = ez*np.sin(t)   # 1M parametric points on the ellipse 

    p = e[np.sum(np.square(e-c), 1).argmin()]   # point on ellipse closest to c 
    return p 



class X(object):
    def __init__(self, root):
        self.root = root 

    def __repr__(self):
        return "\n".join( map(repr, self.constituents()))

    def find(self, shape):
        return self.root.find(shape) 

    def find_one(self, shape):
        ff = self.root.find(shape)
        assert len(ff) == 1
        return ff[0]  

    def constituents(self):
        return self.root.constituents() 


    def replacement_cons(self):
        """
        """ 
        i = self.find_one("STorus")
        r = i.param[0]
        R = i.param[1]

        d = self.find_one("SEllipsoid")
        ex = d.param[0]
        ez = d.param[1]

        print("r %s R %s ex %s ez %s " % (r,R,ex,ez))
        print(" SEllipsoid d.xy %s " % repr(d.xy) ) 
        print(" STorus     i.xy %s " % repr(i.xy) ) 

        z0 = i.xy[1]   # torus z-plane in ellipsoid frame

        p = ellipse_closest_approach_to_point( ex, ez, [R,z0] )    # [R,z0] is center of torus circle 

        pr, pz = p    # at torus/ellipse closest point : no guarantee of intersection 
        print(" ellipse closest approach to torus  %s " % repr(p) )
        
        r2 = pr
        r1 = R - r
        mz = (z0 + pz)/2.   # mid-z cone coordinate (ellipsoid frame)
        hz = (pz - z0)/2.   # cons half height 

        f = SCons( "f", [r1,r2,hz] )
        B = np.array( [0, mz] )  

        print(" replacment SCons %s offset %s " % (repr(f),repr(B)))

        return f, B  


    def spawn_rationalized(self):
        """

        ::
        
                   UnionSolid
                   /         \ 
           Ellipsoid          Subtraction
                              /         \
                            Tubs        Torus


                   UnionSolid
                   /         \ 
           Ellipsoid           Cons



        """
        name = self.__class__.__name__

        x = copy.deepcopy(self) 

        # establish expectations for tree
        e = x.find_one("SEllipsoid")
        t = x.find_one("STorus")
        ss = t.parent
        assert ss is not None and ss.shape == "SSubtractionSolid"
        us = ss.parent  
        assert us is not None and us.shape == "SUnionSolid"
        assert us.left is not None and us.left == e and us.right == ss and ss.right == t
        assert us.right is not None and us.right == ss 


        if name == "x018":   # cathode vacuum cap
            assert x.root.shape == "SIntersectionSolid"
            x.root = e 
            e.parent = None
        elif name == "x019":  # remainder vacuum 
            assert x.root.shape == "SSubtractionSolid"
            left = x.root.left 
            assert left.shape == "SUnionSolid" 
            left.parent = None 
            x.root = left 
        else:
            pass
        pass

        if name in ["x019","x020","x021"]: 
            # calculate the parameters of the replacement cons
            cons, offset =  x.replacement_cons()

            # tree surgery : replacing the right child of UnionSolid  
            us.right = cons
            cons.parent = us
            cons.ltransform = offset 
        pass

        return x 

        

class Shape(object):
    """
    matplotlib patches do not support deferred placement it seems, 
    so do that here 
    """
    KWA = dict(fill=False)
    dtype = np.float64

    PRIMITIVE = ["SEllipsoid","STubs","STorus", "SCons", "SHype", "SBox", "SPolycone"]
    COMPOSITE = ["SUnionSolid", "SSubtractionSolid", "SIntersectionSolid"]

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


    is_primitive = property(lambda self:self.left is None and self.right is None)
    is_composite = property(lambda self:self.left is not None and self.right is not None)

    def _get_xy(self):
        """
        Assumes only translations, adds the node.ltransform obtained by following 
        parent links up the tree of shapes. 



                           a                                            Intersection
                                                                       /            \
                      b             m(D)                          Union             m:Tubs
                                                                  /    \  
                c          k(C)                             Union       Tubs
                                                           /     \
            d     f(B)                            Ellipsoid   Subtraction    
                                                              /          \ 
                g(B)  i(B+A)                                Tubs         Torus


        """
        xy = np.array([0,0], dtype=self.dtype )
        node = self
        while node is not None:
            if node.ltransform is not None:
                log.debug("adding ltransform %s " %  node.ltransform)
                xy += node.ltransform
            pass
            node = node.parent 
        pass
        return xy 

    xy = property(_get_xy)

    def constituents(self):
        if self.is_primitive:
            return [self]
        else: 
            assert self.is_composite
            cts = []
            cts.extend( self.left.constituents() )
            cts.extend( self.right.constituents() )
            return cts
        pass

    def find(self, shape):
        cts = self.constituents()
        return filter( lambda ct:ct.shape == shape, cts ) 

    def patches(self):
        """
        Positioning is relying on self.xy of the primitives
        with nothing being passed into composites.

        For composites self.param[2] is the local right transform
        """
        if self.shape == "SEllipsoid":
            return self.make_ellipse( self.xy, self.param, **self.kwa )
        elif self.shape == "STubs":
            return self.make_rect( self.xy, self.param, **self.kwa)
        elif self.shape == "STorus":
            return self.make_torus( self.xy, self.param, **self.kwa)
        elif self.shape == "SCons":
            return self.make_cons( self.xy, self.param, **self.kwa)
        elif self.shape == "SHype":
            return self.make_hype( self.xy, self.param, **self.kwa)
        elif self.shape == "SBox":
            return self.make_rect( self.xy, self.param, **self.kwa)
        elif self.shape == "SPolycone":
            return self.make_polycone( self.xy, self.param, **self.kwa)
        else:
            assert self.is_composite 
            pts = []
            pts.extend( self.left.patches() )
            pts.extend( self.right.patches() )
            return pts 
        pass

    @classmethod
    def create(cls, pt ):
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
            path_data.append( (act, (vtx[0]+xy[0], vtx[1]+xy[1])) )
        pass
        path_data.append( (Path.CLOSEPOLY, (vtxs[0,0]+xy[0], vtxs[0,1]+xy[1])) )
        pass

        codes, verts = zip(*path_data)
        path = Path(verts, codes)
        patch = PathPatch(path, **kwa)  
        return [patch]

    @classmethod
    def make_cons(cls, xy , param, **kwa ):
        """
        (-r2,z2)      (r2,z2)
              1---------2       
               \       /
                0 ... 3   
        (-r1,z1)     (r1,z1)
        """
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
    def make_polycone(cls, xy , param, **kwa ):
        """
        """
        zp = param
        nz = len(zp)
        assert zp.shape == (nz, 3), zp 
        assert nz > 1 , zp

        rmin = zp[:,0]
        rmax = zp[:,1]
        z = zp[:,2]

        vtxs = np.zeros( (2*nz,2) )
        for i in range(nz):
            vtxs[i] = ( -rmax[i], z[i] )
            vtxs[2*nz-i-1] = ( rmax[i], z[i] )
        pass 
        log.debug(" xy : %r " %  xy )
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


class SEllipsoid(Shape):pass
class STubs(Shape):pass
class STorus(Shape):pass
class SCons(Shape):pass
class SHype(Shape):pass
class SPolycone(Shape):pass
class SUnionSolid(Shape):pass
class SSubtractionSolid(Shape):pass
class SIntersectionSolid(Shape):pass



if __name__ == '__main__':
    pass







