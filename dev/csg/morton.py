#!/usr/bin/env python
"""
Morton Codes
==============


* https://en.wikipedia.org/wiki/Z-order_curve

Ericson RTCD p314
-------------------

Given the locational code for the parent node, a new locational code for one of
its child nodes is easily constructed by left-shifting the parent key by 3 and

    childKey = (parentKey << 3) + childIndex


Morton Codes
--------------

* http://asgerhoedt.dk/?p=276


"""

import numpy as np

bin_ = lambda _:"{0:08b}".format(_)


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

def MortonDecode3D(c):
    assert 0, "3D needs debug"
    z = CompactBy1(c);
    y = CompactBy1(c >> 1);
    x = CompactBy1(c >> 2);
    return x, y, z 
 
  

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

    pchild      = property(lambda self:self.loc & ((1 << self.dim) - 1))   # bottom bits 

    nchild     = property(lambda self:1 << self.dim)  # 2d:4 3d:8
    firstchild = property(lambda self:(self.loc << self.dim) ) 
    lastchild  = property(lambda self:(self.loc << self.dim) | (self.nchild - 1) ) 

    ngrandchild  = property(lambda self:1 << (2*self.dim))  # 2d:4 3d:8
    firstgrandchild = property(lambda self:(self.loc << (2*self.dim)) ) 
    lastgrandchild  = property(lambda self:(self.loc << (2*self.dim)) | (self.ngrandchild - 1) ) 


    def _get_ijk(self):
        if self.dim == 2:
            i,j = MortonDecode2D(self.loc)
            k = 0 
        else:
            i,j,k = MortonDecode3D(self.loc)
        pass
        return Ijk((i,j,k))
    ijk = property(_get_ijk)


    @classmethod 
    def FromIjk(cls, ijk, dim=2):
        assert dim in (2,3) 
        c = MortonCode2D(ijk[0],ijk[1]) if dim == 2 else MortonCode3D(ijk[0],ijk[1],ijk[2])
        loc = Loc(c, dim=dim)
        ijk2 = loc.ijk
        assert ijk2 == ijk
        return loc


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



def test_2d(n, limit=16):
    nn = n*n
    print "test_2d n:%d nn:%d " % ( n, nn) 
    cc = range(nn) if nn < 2*limit else range(0,limit) + range(nn-limit, nn)
    for c in cc:
        m = Loc(c)
        ijk = m.ijk
        print " %r %r " % ( m, ijk )

def test_3d():
    print "test_3d" 
    for c in range(16):
        m = Loc(c,dim=3)
        ijk = m.ijk
        print " %r %r " % ( m, ijk )




if __name__ == '__main__':

    test_2d(1)
    test_2d(2)
    test_2d(4)
    test_2d(8)
    #test_3d()


