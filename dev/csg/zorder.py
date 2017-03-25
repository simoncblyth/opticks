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






class Loc(object):
    """
    NB parent, grandparent, firstchild, lastchild are morton codes
       within different resolution sets 

    Do Morton code works across all resolutions ? 
  
    Nope they are recycled at every level, so need to incorporate
    the level into the key for absolute cross level addressing.

    """
    def __init__(self, loc, dim=2, width=16):
        self.loc = loc  
        self.dim = dim
        self.width = width  # presentation field width

    parent      = property(lambda self:self.loc >> self.dim)
    grandparent = property(lambda self:self.loc >> (2*self.dim))

    pchild      = property(lambda self:self.loc & ((1 << self.dim) - 1))

    nchild     = property(lambda self:1 << self.dim)  # 2d:4 3d:8
    firstchild = property(lambda self:(self.loc << self.dim) ) 
    lastchild  = property(lambda self:(self.loc << self.dim) | (self.nchild - 1) ) 


    def child(self, c_loc):
        return (self.loc << self.dim) | c_loc

    def __repr__(self):
        loc = self.loc
        dim = self.dim

        bfmt = "{0:0%db}" % self.width
        b_ = lambda _:bfmt.format(_)

        _loc = b_(loc)
        _parent = b_(self.parent )
        _grandparent = b_( self.grandparent )
        _firstchild = b_( self.firstchild )
        _lastchild  = b_( self.lastchild )
        _pchild = "%d" % self.pchild

        return "%7d : loc %s pch:%s par: %s  fc: %s lc: %s " % (loc, _loc,_pchild,  _parent, _firstchild, _lastchild  )

class Ijk(object):
    def __init__(self, ijk, dim=2):
       self.ijk = np.asarray(ijk, dtype=np.uint32)
       self.dim = dim

    i = property(lambda self:self.ijk[0]) 
    j = property(lambda self:self.ijk[1]) 
    k = property(lambda self:self.ijk[2]) 

    def __repr__(self):
        return self.desc(self.ijk, self.dim) 
    @classmethod
    def desc(cls, ijk, dim=2):
        b4_ = lambda _:"{0:04b}".format(_)
        x_ = lambda _:"{0:01x}".format(_)
        _ijk = "(%2d,%2d,%2d) " % tuple(ijk)
        _ijk_b = "(%4s,%4s,%4s) " % tuple(map(b4_, ijk))
        _ijk_x = "(%s,%s,%s) " % tuple(map(x_, ijk))
        return " %s %s %s " % ( _ijk, _ijk_b, _ijk_x )

class Xyz(object):
    def __init__(self, xyz, dim=2):
       self.xyz = np.asarray(xyz, dtype=np.float32)
       self.dim = dim

    x = property(lambda self:self.xyz[0]) 
    y = property(lambda self:self.xyz[1]) 
    z = property(lambda self:self.xyz[2]) 
    xy = property(lambda self:(self.x, self.y, 0.))

    def __repr__(self):
        return self.desc(self.xyz, self.dim) 
    @classmethod
    def desc(cls, xyz, dim=2):
        _xyz = "(%5.2f,%5.2f,%5.2f)" % tuple(xyz)
        return " %s " % _xyz 


class Point(object):
    def __init__(self, pos):
        assert self.domain, "must externally set the domain"

        xyz = Xyz(pos)
        ijk = self.domain.xyz2ijk(xyz.xyz)
        loc = self.domain.ijk2loc(ijk.ijk)

        self.xyz = xyz
        self.ijk = ijk
        self.loc = loc

    x = property(lambda self:self.xyz.x) 
    y = property(lambda self:self.xyz.y) 
    z = property(lambda self:self.xyz.z) 
    xy = property(lambda self:(self.x, self.y, 0.))

    i = property(lambda self:self.ijk.x) 
    j = property(lambda self:self.ijk.y) 
    k = property(lambda self:self.ijk.z) 

    def __repr__(self):
        return "P  %r %r %r " % (self.xyz, self.ijk, self.loc )




class BBox(object):
    def __init__(self, bb):
        bb = np.asarray(bb, dtype=np.float32)
        self.bb = bb
        self.origin = bb[0]
        self.side  = bb[1] - bb[0]

    min = property(lambda self:self.bb[0])
    max = property(lambda self:self.bb[1])

    def pos(self, fr):
        return self.origin + self.side*fr 

    def scaled(self, p):
        """
        :param p: global xyz float coordinate within bbox
        :return sc: bbox offset/scaled coordinates 

        positions outside the bbox will yield values outside 0:1
        """ 
        return (np.asarray(p) - self.origin)/self.side

    def set_lim(self, ax, margin=1):
        ax.set_xlim(self.min[X] - margin, self.max[X] + margin)
        ax.set_ylim(self.min[Y] - margin, self.max[Y] + margin)



class Domain(object):
    """
    Most of this doesnt depend on the level, so split ?
    """
    def __init__(self, bb, level=0, dim=2, maxlevel=16):
        assert level > -1 and level < maxlevel
        hijk = np.zeros( (maxlevel, 3), dtype=np.uint32)
        for h in range(maxlevel):
            hijk[h] = [1 << h,1 << h,1 << h if dim == 3 else 1]
        pass
        self.hijk = hijk
        self.level = level
        self.dim = dim

        self.bb = bb

    def spawn_domain(self, level):
        return Domain(self.bb, level=level )

    def __repr__(self):
        return " h %2d nloc %8d nijk %15r bb %r %r sxyz %r " % (self.level, self.nloc, tuple(self.nijk), tuple(self.bb.min), tuple(self.bb.max), tuple(self.sxyz) )  

    def frac(self, ijk):
        # depends on level from nijk
        assert ijk is not None
        ijk = ijk.ijk if type(ijk) is Ijk else ijk
        fijk = np.asarray(ijk, dtype=np.float32)
        return fijk/self.nijk

    def ijk2xyz(self, ijk):
        # depends on level from frac 
        fr = self.frac(ijk)
        pos = self.bb.pos(fr) 
        return Xyz(pos, dim=self.dim)

    nchild = property(lambda self:1 << self.dim )
    nijk = property(lambda self:self.hijk[self.level])
    nloc = property(lambda self:1 << (self.dim*self.level))
    sxyz = property(lambda self:self.bb.side/self.nijk)
    hxyz = property(lambda self:self.sxyz/2.)

    def xyz2ijk(self, p):
        """
        depends on level from nijk

        :param p:
        :return ijk: integer coordinates
        """
        sc = self.bb.scaled(p)
        ijk = np.array( self.nijk*sc, dtype=np.uint32 )
        return Ijk(ijk, dim=self.dim)

    def loc2ijk(self, c):
        #NB doesnt depend on level
        if self.dim == 2:
            i,j = MortonDecode2D(c)
        else:
            assert 0
        pass
        k = 0 
        return Ijk((i,j,k))

    def ijk2loc(self, ijk):
        #NB doesnt depend on level
        i,j,k = ijk.ijk if type(ijk) is Ijk else ijk
        if self.dim == 2:
            c = MortonCode2D(i,j)
            ijk2 = self.loc2ijk(c)
            assert ijk2.i == i
            assert ijk2.j == j
        elif self.dim == 3:
            c = MortonCode3D(i,j,k)
        else:
            assert 0
        pass
        return Loc(c, dim=self.dim)

    def loc2xyz_ijk(self, loc):
        ijk = self.loc2ijk(loc)
        xyz = self.ijk2xyz(ijk)
        return xyz,ijk 

    def corners(self, loc):
        """
        Should this be halfing ? using hxyz or not sxyz
        """
        xyz, ijk = self.loc2xyz_ijk(loc)
        nc = 1 << self.dim
        cnrs = np.zeros( (nc, 3), dtype=np.float32 )
        for ic in range(nc):
            off = self.loc2ijk(ic)  # (0,0) (1,0) (0,1) (1,1)
            cnr = xyz.xyz + off.ijk*self.sxyz
            cnrs[ic] = cnr
        pass
        return cnrs

    def __call__(self, loc, node):
        cnrs = self.corners(loc) 
        corners = 0 
        for ic,cnr in enumerate(cnrs):
            sdf = node(cnr)
            inside = sdf < 0.
            corners |= (inside << ic) 
        pass
        return corners

    def zorder_dump(self, levels, limit=16):
        keep_level = self.level
        for level in levels:
            self.level = level
            self._zorder_dump(limit=limit)
        pass
        self.level = keep_level 

    def _zorder_dump(self, limit=16):
        nloc = self.nloc 
        print self
        cc = range(nloc)
        if nloc > 2*limit:
            cc = range(limit) + range(nloc-limit, nloc) 
        else:
            cc = range(nloc)
        pass
        for c in cc:
            loc = Loc(c, width=(self.level+1)*self.dim)  # +1 as includes child loc
            xyz,ijk = self.loc2xyz_ijk(c)
            print " %r %r %r " % ( loc, ijk, xyz )
        pass

    def zorder_plot(self, ax):
        nloc = self.nloc 
        for loc in range(nloc-1):
            p0,ijk0 = self.loc2xyz_ijk(loc)
            p1,ijk1 = self.loc2xyz_ijk(loc+1)
            ax.plot( [p0.x,p1.x], [p0.y,p1.y]) 
        pass


class Corners(object):
    """
    ::

         2   3
         
         0   1

    """
    def __init__(self, ax, cnrs):
        self.ax = ax
        self.cnrs = cnrs

    def plot_edge(self, ia, ib, col):
        a = self.cnrs[ia]
        b = self.cnrs[ib]
        self.ax.plot( [a[X],b[X]], [a[Y],b[Y]], col) 

    def plot(self, col):
        self.plot_edge(0,1,col)
        self.plot_edge(1,3,col)
        self.plot_edge(3,2,col)
        self.plot_edge(2,0,col)


class Node(object):
    def __init__(self, key, corners=None, dim=2):
         self.key = key 
         self.corners = corners
         self.children = [None for _ in range(1 << dim)]

    level = property(lambda self:self.key[0])
    loc = property(lambda self:self.key[1])

    def __repr__(self):
         _corners = "cnrs: %x" % self.corners if self.corners is not None else ""
         _loc = Loc(self.loc)
         return "[%2d] %r %s " % (self.level, _loc, _corners) 

    def postorder(self):
         nodes = []
         def traverse_r(node, depth=0):
             assert node.level == depth 
             for child in filter(None, node.children):
                 traverse_r(child, depth=depth+1) 
             pass 
             nodes.append(node)
         pass
         traverse_r(self)
         return nodes

    def corners_plot(self, ax, doms):
        postorder = self.postorder()
        cols = 'rgbcmykkkkkkk'
        for node in postorder:
            level, loc = node.key
            cnrs = Corners(ax,doms[level].corners(loc))
            cnrs.plot(col=cols[level])
        pass



if __name__ == '__main__':
    pass

    plt.ion()
    plt.close("all")
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1, aspect='equal')

    bb = BBox([[0.,0.,0.],[80.,80.,80.]])
    bb.set_lim(ax)

    root = CSG("sphere", param=[40.,40.,0,20.])

    rdr = Renderer(ax)
    rdr.render(root)
    
    level = 7
    dim = 2
    maxcorner = (1 << (1 << dim)) - 1 # 2d:0xf  3d:0xff  one bit for each child
    msk = (1 << dim) - 1              # 2d:0b11 3d:0b111 one bit for each dimension

    doms = []
    for lev in range(level+1):
        dom = Domain(bb, level=lev)
        doms.append(dom)
    pass
    print "\n".join(map(repr,doms))

    domain = doms[level]           # leaf level is special
    #domain.zorder_dump(range(10))
    domain.zorder_plot(ax)


    ## morton zorder traverse at level
    ## collecting the leaves 

    lquad = {}
    for c in range(domain.nloc):
        corners = domain(c, root)
        if corners > 0 and corners < maxcorner:
            key = (domain.level, c)
            lquad[key] = Node(key, corners)
        pass
    pass 
    nleaf = len(lquad)

    ## iterative bottom up quadtree construction, 
    ## pulling up from the leaves

    for key,leaf in lquad.items():
        level, loc = key

        node = leaf 

        uloc = loc
        uchild = loc & msk 

        for elevation in range(1, level+1):
            ulev = level - elevation   

            uloc >>= domain.dim         #  level by level relative
            uloc2 = loc >> (domain.dim*elevation)  # absolute 
            assert uloc2 == uloc

            ukey = (ulev,uloc)
         
            if ukey not in lquad:
                unode = Node(ukey) 
                lquad[ukey] = unode
            else:
                unode = lquad[ukey]
            pass
            unode.children[uchild] = node 
            pass
            node = unode           # hold on to prior node
            uchild = uloc & msk   # uchild index is from the lower level, so update in tail
        pass
    pass        

    top = node
    nodes = top.postorder()
    nnode = len(nodes)

    print "nleaf %d nnode %d " % (nleaf, nnode) 

    top.corners_plot(ax, doms)


    #ax.axis('auto') 
    fig.show()



