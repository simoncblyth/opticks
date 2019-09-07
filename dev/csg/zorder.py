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

* https://en.wikipedia.org/wiki/Z-order_curve


Even 2d Grids get real big real soon
---------------------------------------

::

    In [4]: run zorder.py
     h  0 nloc            1 nijk            (1, 1, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (80.0, 80.0, 80.0) 
     h  1 nloc            4 nijk            (2, 2, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (40.0, 40.0, 80.0) 
     h  2 nloc           16 nijk            (4, 4, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (20.0, 20.0, 80.0) 
     h  3 nloc           64 nijk            (8, 8, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (10.0, 10.0, 80.0) 
     h  4 nloc          256 nijk          (16, 16, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (5.0, 5.0, 80.0) 
     h  5 nloc         1024 nijk          (32, 32, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (2.5, 2.5, 80.0) 
     h  6 nloc         4096 nijk          (64, 64, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (1.25, 1.25, 80.0) 
     h  7 nloc        16384 nijk        (128, 128, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.625, 0.625, 80.0) 
     h  8 nloc        65536 nijk        (256, 256, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.3125, 0.3125, 80.0) 
     h  9 nloc       262144 nijk        (512, 512, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.15625, 0.15625, 80.0) 
     h 10 nloc      1048576 nijk      (1024, 1024, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.078125, 0.078125, 80.0) 
     h 11 nloc      4194304 nijk      (2048, 2048, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.0390625, 0.0390625, 80.0) 
     h 12 nloc     16777216 nijk      (4096, 4096, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.01953125, 0.01953125, 80.0) 
     h 13 nloc     67108864 nijk      (8192, 8192, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.009765625, 0.009765625, 80.0) 
     h 14 nloc    268435456 nijk    (16384, 16384, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.0048828125, 0.0048828125, 80.0) 
     h 15 nloc   1073741824 nijk    (32768, 32768, 1) bb (0.0, 0.0, 0.0) (80.0, 80.0, 80.0) sxyz (0.00244140625, 0.00244140625, 80.0) 
    nleaf 252 nnode 485 


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
from morton import Loc, Ijk


X,Y,Z,W = 0,1,2,3

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



class Grid(object):
    """
    Most of this doesnt depend on the level, so split ?
    """
    maxlevel = 16 

    def __init__(self, bb, level=0, dim=2):
        assert level > -1 and level < self.maxlevel
        hijk = np.zeros( (self.maxlevel, 3), dtype=np.uint32)
        for h in range(self.maxlevel):
            hijk[h] = [1 << h,1 << h,1 << h if dim == 3 else 1]
        pass
        self.hijk = hijk
        self.level = level
        self.dim = dim
        pass
        self.bb = bb

    def __repr__(self):
        return " h %2d nloc %12d nijk %20r bb %r %r sxyz %r " % (self.level, self.nloc, tuple(self.nijk), tuple(self.bb.min), tuple(self.bb.max), tuple(self.sxyz) )  

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


    def loc2xyz_ijk(self, c):
        loc = Loc(c) 
        ijk = loc.ijk 
        xyz = self.ijk2xyz(ijk)
        return xyz,ijk 

    def corners(self, c):
        """
        Should this be halfing ? using hxyz or not sxyz
        """
        loc = Loc(c)
        ijk = loc.ijk
        xyz = self.ijk2xyz(ijk)

        nc = 1 << self.dim
        cnrs = np.zeros( (nc, 3), dtype=np.float32 )
        for ic in range(nc):
            loc = Loc(ic)
            off = loc.ijk  # (0,0) (1,0) (0,1) (1,1)
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

    def corners_plot(self, ax, grids):
        postorder = self.postorder()
        cols = 'rgbcmykkkkkkk'
        for node in postorder:
            level, loc = node.key
            cnrs = Corners(ax,grids[level].corners(loc))
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


    grids = []
    for lev in range(Grid.maxlevel):
        g = Grid(bb, level=lev)
        grids.append(g)
    pass
    print "\n".join(map(repr,grids))



    grid = grids[level]           # leaf level is special
    #grid.zorder_dump(range(10))
    grid.zorder_plot(ax)


    ## morton zorder traverse at level
    ## collecting the leaves 

    lquad = {}
    for c in range(grid.nloc):
        corners = grid(c, root)
        if corners > 0 and corners < maxcorner:
            key = (grid.level, c)
            lquad[key] = Node(key, corners)
        pass
    pass 
    nleaf = len(lquad)

    ## iterative bottom up quadtree construction, 
    ## pulling up from the leaves

    first = True

    for key,leaf in lquad.items():
        level, loc = key

        node = leaf 
        dchild = loc & msk 
        dloc = loc
        depth = level - 1

        while depth >= 0:
            dloc >>= grid.dim
            dkey = (depth, dloc) 

            if dkey not in lquad:
                dnode = Node(dkey)
                lquad[dkey] = dnode
            else:
                dnode = lquad[dkey]
            pass 
            dnode.children[dchild] = node 
            node = dnode

            dchild = dloc & msk   # dchild updated in tail
            depth -= 1 
        pass

        #uloc = loc
        #uchild = loc & msk 
        #for elevation in range(1, level+1):
        #    ulev = level - elevation   
        #
        #    uloc >>= grid.dim         #  level by level relative
        #    uloc2 = loc >> (grid.dim*elevation)  # absolute 
        #    assert uloc2 == uloc
        #
        #    ukey = (ulev,uloc)
        # 
        #    if ukey not in lquad:
        #        unode = Node(ukey) 
        #        lquad[ukey] = unode
        #    else:
        #        unode = lquad[ukey]
        #    pass
        #    unode.children[uchild] = node 
        #    pass
        #    node = unode           # hold on to prior node
        #    uchild = uloc & msk   # uchild index is from the lower level, so update in tail
        #pass
    pass        

    top = node
    nodes = top.postorder()
    nnode = len(nodes)

    print "nleaf %d nnode %d " % (nleaf, nnode) 

    top.corners_plot(ax, grids)


    #ax.axis('auto') 
    fig.show()



