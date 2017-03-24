#!/usr/bin/env python
"""

* https://en.wikipedia.org/wiki/Z-order_curve


Ericson RTCD p314
-------------------

Given the locational code for the parent node, a new locational code for one of
its child nodes is easily constructed by left-shifting the parent key by 3 and

    childKey = (parentKey << 3) + childIndex



Morton Codes
--------------

* http://asgerhoedt.dk/?p=276

Quadtrees
----------


PARALLEL CONSTRUCTION OF QUADTREES AND QUALITY TRIANGULATIONS

* http://www.worldscientific.com/doi/abs/10.1142/S0218195999000303


"""

from collections import OrderedDict as odict 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 18,10.2 

from csg import CSG 
from nodeRenderer import Renderer


X,Y,Z,W = 0,1,2,3


def SeparateBy1(x):
    """
    http://asgerhoedt.dk/?p=276
    """
    x &= 0x0000ffff;                  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x <<  8)) & 0x00ff00ff; # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x <<  4)) & 0x0f0f0f0f; # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x <<  2)) & 0x33333333; # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x <<  1)) & 0x55555555; # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x

def MortonCode2D(x, y):
    return SeparateBy1(x) | (SeparateBy1(y) << 1);


def CompactBy1(x):
    x &= 0x55555555;                  #// x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >>  1)) & 0x33333333; #// x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >>  2)) & 0x0f0f0f0f; #// x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >>  4)) & 0x00ff00ff; #// x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >>  8)) & 0x0000ffff; #// x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;

def MortonDecode2D(c):
    x = CompactBy1(c);
    y = CompactBy1(c >> 1);
    return x, y
  

def expandBits(v):
    """
    https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
    """
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v;

def MortonCode3D(ix, iy, iz):
    # 30-bit Morton code,  (ix,iy,iz) must be in 0 to 1023, 1 << 10 = 2^10 = 1024
    xx = expandBits(ix)
    yy = expandBits(iy)
    zz = expandBits(iz)
    return (xx << 2) + (yy << 1) + zz


class Domain(object):
    def __init__(self, hh, bb, dim=2):
        hijk = np.zeros( (len(hh), 3), dtype=np.uint32)
        for h in hh:
            ni = 1 << h 
            nj = 1 << h 
            nk = 0
            hijk[h] = [ni,nj,nk]
        pass
        self.hijk = hijk
        self.h = 0
        self.dim = dim

        bb = np.asarray(bb, dtype=np.float32)

        self.bb = bb
        self.origin = bb[0]
        self.side  = bb[1] - bb[0]

    def scaled(self, p):
        """
        :param p: global xyz float coordinate within bbox
        :return sc: bbox offset/scaled coordinates 

        positions outside the bbox will yield values outside 0:1
        """ 
        return (np.asarray(p) - self.origin)/self.side

    def frac(self, ijk):
        fijk = np.asarray(ijk, dtype=np.float32)
        return fijk/self.nijk

    def xyz(self, ijk):
        fr = self.frac(ijk)
        return self.origin + self.side*fr 

    nijk = property(lambda self:self.hijk[self.h])


    def ijk(self, p):
        """
        :param p:
        :return ijk: integer coordinates
        """
        sc = self.scaled(p)
        return np.array( self.nijk*sc, dtype=np.uint32 )

    def code(self, ijk):
        i,j,k = ijk
        if self.dim == 2:
            c = MortonCode2D(i,j)
            i2,j2 = MortonDecode2D(c)
            assert i2 == i
            assert j2 == j
        elif self.dim == 3:
            c = MortonCode3D(i,j,k)
        else:
            assert 0
        pass
        return c

    def set_lim(self, ax, margin=1):
        ax.set_xlim(self.bb[0][X] - margin, self.bb[1][X] + margin)
        ax.set_ylim(self.bb[0][Y] - margin, self.bb[1][Y] + margin)



class Point(object):
    def __init__(self, pos):
        assert self.domain, "must externally set the domain"
        self.pos = pos

        ijk = self.domain.ijk(pos)
        loc = self.domain.code(ijk)

        self.ijk = ijk
        self.loc = loc

    def __repr__(self):

        b4_ = lambda _:"{0:04b}".format(_)
        b8_ = lambda _:"{0:08b}".format(_)
        x_ = lambda _:"{0:01x}".format(_)

        ijk = self.ijk
        loc = self.loc
        dim = self.domain.dim

        _ijk = "(%2d,%2d,%2d) " % tuple(ijk)
        _ijk_b = "(%4s,%4s,%4s) " % tuple(map(b4_, ijk))
        _ijk_x = "(%s,%s,%s) " % tuple(map(x_, ijk))

        _loc = b8_(loc)
        _loc_parent = b8_(loc >> dim)
        _loc_grandparent = b8_(loc >> (2*dim))

        return "P  %s %s %s  morton %s     m-parent: %s  m-grandpa: %s " % (_ijk, _ijk_b, _ijk_x, _loc, _loc_parent, _loc_grandparent )


if __name__ == '__main__':
    pass

    plt.ion()
    plt.close("all")
    fig = plt.figure()

    nh = 7
    ax = fig.add_subplot(1,1,1, aspect='equal')

    bb = [[-10.,-10.,-10.],[10.,10.,10.]]

    root = CSG("sphere", param=[0,0,0,7])


    rdr = Renderer(ax)
    rdr.render(root)
    
    hh = range(nh)
 
    domain = Domain(hh, bb ) 
    Point.domain = domain 

    d = odict()

    # collecting points on multiple resolution rasters into the odict 
    for h in hh:
        domain.h = h     # setting domain resolution 
        nijk = domain.nijk
        ni,nj,nk = nijk
        print "------------- h %d  %s  " % ( h, repr(nijk) )
        for i in range(ni):
            for j in range(nj):
                pos = domain.xyz((i,j,0))
                pos[2] = 0 
                p = Point(pos) 
                d[p.loc] = p
            pass
        pass
    pass

    # morton code works across all resolutions ?
    zorder = sorted(d)
    for l in range(len(zorder)-1):
        c0 = zorder[l]
        c1 = zorder[l+1]
        p0 = d[c0]
        p1 = d[c1]
        sdf0 = root(p0.pos)
        sdf1 = root(p1.pos)

        if np.sign(sdf0) != np.sign(sdf1):
            x0 = p0.pos[X]
            y0 = p0.pos[Y]
            x1 = p1.pos[X]
            y1 = p1.pos[Y]
            ax.plot( [x0,x1], [y0,y1]) 
        pass
        domain.set_lim(ax)
    pass

    # TODO: build a quadtree with leafs where squares have sdf sign action 
    #       use it to approximate the circle with lines


    #ax.axis('auto') 
    fig.show()

     





