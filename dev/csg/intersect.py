#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from node import Node, SPHERE, BOX, EMPTY, desc_sh
from node import root0, root1, root2, root3, root4
from node import UNION, INTERSECTION, DIFFERENCE


import logging
log = logging.getLogger(__name__)

f_ = lambda v:"%5.2f" % v if v is not None else -1



class II(np.ndarray):
    """
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    Enable a numpy float32 array to act like a muggle object 
    with a mix of float and uint props that are actually 
    stored in a simple float32 array, allowing collections
    and operations on millions of intersects.

    ::

        0,0  normal.x
        0,1  normal.y
        0,2  normal.z
        0,3  t 

        1,0  tmin       (f32)
        1,1  idx : node.idx   (u32)
        1,2  node : shape or operation (u32)
        1,3  seq

        2,0  ray.origin.x
        2,1  ray.origin.y
        2,2  ray.origin.z
        2,3  rtmin : resulting tmin 

        3,0  ray.direction.x
        3,1  ray.direction.y
        3,2  ray.direction.z
        3,3  

    """

    N_ = 0, slice(0,3)
    T_ = 0, 3

    TMIN_ = 1, 0
    IDX_ = 1, 1 
    NODE_ = 1, 2
    SEQ_ = 1, 3

    O_ = 2, slice(0,3)
    RTMIN_ = 2, 3

    D_ = 3, slice(0,3)


    def __new__(cls, a=None, history=[]):
        if a is None:
           a = np.zeros((4,4), dtype=np.float32 )
        pass
        obj = np.asarray(a).view(cls)
        obj.history = history
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.history = getattr(obj, 'history', None)


    def _get_t(self):
        return self[self.T_[0],self.T_[1]]
    def _set_t(self, t):
        self[self.T_[0],self.T_[1]] = t 
    t = property(_get_t, _set_t)

    def _get_tmin(self):
        return self[self.TMIN_[0],self.TMIN_[1]]
    def _set_tmin(self, t):
        self[self.TMIN_[0],self.TMIN_[1]] = t 
    tmin = property(_get_tmin, _set_tmin)

    def _get_rtmin(self):
        return self[self.RTMIN_[0],self.RTMIN_[1]]
    def _set_rtmin(self, t):
        self[self.RTMIN_[0],self.RTMIN_[1]] = t 
    rtmin = property(_get_rtmin, _set_rtmin)

    def _get_n(self):
        return self[self.N_[0],self.N_[1]]
    def _set_n(self, n):
        self[self.N_[0],self.N_[1]] = n 
    n = property(_get_n, _set_n)

    def _get_o(self):
        return self[self.O_[0],self.O_[1]]
    def _set_o(self, o):
        self[self.O_[0],self.O_[1]] = o 
    o = property(_get_o, _set_o)

    def _get_d(self):
        return self[self.D_[0],self.D_[1]]
    def _set_d(self, o):
        self[self.D_[0],self.D_[1]] = o 
    d = property(_get_d, _set_d)

    def _get_idx(self):
        return self.view(np.uint32)[self.IDX_[0],self.IDX_[1]]
    def _set_idx(self, u):
        self.view(np.uint32)[self.IDX_[0],self.IDX_[1]] = u 
    idx = property(_get_idx, _set_idx)

    def _get_node(self):
        return self.view(np.uint32)[self.NODE_[0],self.NODE_[1]]
    def _set_node(self, u):
        self.view(np.uint32)[self.NODE_[0],self.NODE_[1]] = u 
    node = property(_get_node, _set_node)


    def _get_seq(self):
        return self.view(np.uint32)[self.SEQ_[0],self.SEQ_[1]]
    def _set_seq(self, u):
        self.view(np.uint32)[self.SEQ_[0],self.SEQ_[1]] = u 
    seq = property(_get_seq, _set_seq)

    xseq = property(lambda self:"%x" % self.seq)

    def _get_seqslot(self):
        """
        Return next available 4bit slot in the sequence
        """
        q = self.seq
        n = 0 
        while (( q & (0xf << 4*n)) >> 4*n ): n += 1
        return n 
    seqslot = property(_get_seqslot)

    def addseq(self, v):
        """
        Adding 4 bits at a time to form a sequence of codes in range 0x1-0xf
        """
        assert v <= 0xf
        q = self.seq
        n = self.seqslot
        assert n*4 <= 32, "addseq overflow assuming 32 bit dtype %d " % n*4

        q |= ((0xf & v) << 4*n) 
        self.seq = q



class IIS(np.ndarray):
    """
    Collection of intersects
    """
    def __new__(cls, a=None, history=[]):
        if a is None:
           a = np.zeros((2,100,4,4), dtype=np.float32 )
        pass
        obj = np.asarray(a).view(cls)
        obj.history = history
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.history = getattr(obj, 'history', None)

    t = property(lambda self:self[:,:,II.T_[0],II.T_[1]])
    n = property(lambda self:self[:,:,II.N_[0],II.N_[1]])
    o = property(lambda self:self[:,:,II.O_[0],II.O_[1]])
    d = property(lambda self:self[:,:,II.D_[0],II.D_[1]])
    tmin = property(lambda self:self[:,:,II.TMIN_[0],II.TMIN_[1]])
    rtmin = property(lambda self:self[:,:,II.RTMIN_[0],II.RTMIN_[1]])

    idx = property(lambda self:self.view(np.uint32)[:,:,II.IDX_[0],II.IDX_[1]])
    node = property(lambda self:self.view(np.uint32)[:,:,II.NODE_[0],II.NODE_[1]])
    seq = property(lambda self:self.view(np.uint32)[:,:,II.SEQ_[0],II.SEQ_[1]])
    # intersect position
    ipos = property(lambda self:self.d * np.repeat(self.t,3).reshape(2,-1,3) + self.o)

    def tpos(self, t):
        return self.d * t + self.o

    def _get_cseq(self):
        """
        Apply index to color code mapping to the seq array
        """
        _cseq = np.zeros( self.seq.shape, dtype=np.string_)
        _cseq[:] = 'k'
        for k, v in self._ctrl_color.items(): 
            _cseq[self.seq == k] = v
        pass
        return _cseq
    cseq = property(_get_cseq)



def test_ii_addseq():
    a = np.zeros((4,4), dtype=np.float32)
    i = II(a)
    i.addseq(0xa)
    i.addseq(0xb)
    i.addseq(0xc)
    assert i.seq == 0xcba 

    i.seq = 0
    i.addseq(0xa)
    i.addseq(0xb)
    assert i.seq == 0xba 
 
    i.seq = 0
    i.addseq(0xa)
    i.addseq(0xb)
    i.addseq(0xc)
    i.addseq(0xf)
    assert i.seq == 0xfcba 
 
    i.seq = 0
    i.addseq(0xa)
    i.addseq(0xb)
    i.addseq(0xc)
    i.addseq(0xe)
    i.addseq(0xf)
    assert i.seq == 0xfecba 
 
    return i 

"""
class I(object):
   def __init__(self, t, n, name="", code=0):
       self.t = t
       self.n = n 
       self.name = name
       self.code = code
       self.history = []

   def __repr__(self):
       return "[%s;%s]" % ( fmt_3f(self.n), fmt_f(self.t))
       #return "I(%s [%s]) %s %d " % ( fmt_f(self.t), fmt_3f(self.n), self.name, self.code )
"""


def intersect_miss(node, ray, tmin):
    a = np.zeros( (4,4), dtype=np.float32 )
    isect = II(a)

    isect.tmin = tmin
    isect.idx = node.idx

    if node.shape is not None:
        isect.node = node.shape
    elif node.operation is not None:
        isect.node = node.operation
    else:
        log.warning("skipped shape for node %s " % node )
        assert 0
        pass

    isect.o = ray.origin
    isect.d = ray.direction

    return isect 
 

def intersect_primitive(node, ray, tmin):
    assert node.is_primitive
    if node.shape == BOX:
        tt, nn = intersect_box( node.param, ray, tmin )  
    elif node.shape == SPHERE:
        tt, nn = intersect_sphere( node.param, ray, tmin)  
    elif node.shape == EMPTY:
        tt, nn = None, None
    else:
        log.fatal("shape unhandled shape:%s desc_shape:%s node:%s " % (node.shape, desc[node.shape], repr(node)))
        assert 0
    pass
    #print " intersect_node %s ray.direction %s tt %s nn %s " % ( desc[shape], repr(ray.direction), tt, repr(nn))

    isect = intersect_miss( node, ray, tmin)
    if tt is not None and nn is not None:
        isect.t = tt
        isect.n = nn
    pass

    isect.history.append("intersect_primitive %s tmin %s tt %s " % (node.tag, f_(tmin), f_(tt) ))
    return isect 




def fmt_f(v):
    return "%5.2f" % (v if v is not None else -1)

def fmt_3f(a):
    sv = ",".join(map(fmt_f,a[:3])) if a is not None else "-"   
    return "(" + sv + ")" 

def intersect_sphere( param, ray, tmin ):

    center = param[:3]
    radius = param[3]

    O = ray.origin - center
    D = ray.direction

    b = np.dot(O, D)
    c = np.dot(O, O) - radius*radius

    disc = b*b - c
    sdisc = np.sqrt(disc) if disc > 0. else 0.

    root1 = -b - sdisc
    root2 = -b + sdisc   ## root2 always > root1


    tt = None
    nn = None

    has_intersect = sdisc > 0.
    if has_intersect:
        #tt = root1 if root1 > tmin else root2 
        if root1 > tmin or root2 > tmin:
            if root1 > tmin:
                tt = root1 
            elif root2 > tmin:
                tt = root2
            else:
                assert 0
            pass 
            nn = (O + tt*D)/radius 
        pass
    pass
    #log.info("intersect_sphere center %s radius %s ray %r -> tt %s nn %s " % ( fmt_3f(center), fmt_f(radius), ray, fmt_f(tt), fmt_3f(nn)))

    return tt, nn


def intersect_box( param, ray, tmin ):

    cen = param[:3]
    sid = param[3]
    bmin = cen - sid
    bmax = cen + sid

    t0 = (bmin - ray.origin)/ray.direction  # intersects with bmin x/y/z slabs
    t1 = (bmax - ray.origin)/ray.direction  # intersects with bmax x/y/z slabs

    near = np.minimum( t0, t1)   # bmin or bmax intersects closest to origin
    far  = np.maximum( t0, t1)   # bmin or bmax intersects farthest from origin

    t_near = np.max(near)    # furthest near intersect
    t_far  = np.min(far)     # closest far intersect

    #log.info("intersect_box tmin %5.2f t_near %5.2f t_far %5.2f " % (tmin,t_near,t_far))
   
    along_x = ray.direction[0] != 0. and ray.direction[1] == 0. and ray.direction[2] == 0. 
    along_y = ray.direction[0] == 0. and ray.direction[1] != 0. and ray.direction[2] == 0. 
    along_z = ray.direction[0] == 0. and ray.direction[1] == 0. and ray.direction[2] != 0. 
 
    in_x = ray.origin[0] > bmin[0] and ray.origin[0] < bmax[0]
    in_y = ray.origin[1] > bmin[1] and ray.origin[1] < bmax[1]
    in_z = ray.origin[2] > bmin[2] and ray.origin[2] < bmax[2]
 
    if along_x:
        has_intersect = in_y and in_z 
    elif along_y:
        has_intersect = in_x and in_z 
    elif along_z:
        has_intersect = in_x and in_y
    else:
        has_intersect = t_far > t_near and t_far > 0.    # segment of ray intersects box, at least one intersect is ahead 


    tt = None
    nn = None
    if has_intersect:
        if tmin < t_near or tmin < t_far:
            if tmin < t_near:
                tt = t_near 
            elif tmin < t_far:
                tt = t_far 
            else:
                assert 0 
            pass
            p = ray.origin + tt*ray.direction - cen
            pa = np.abs(p)
            nn = np.array([0,0,0], dtype=np.float32)
            if   pa[0] >= pa[1] and pa[0] >= pa[2]: nn[0] = np.copysign( 1, p[0] )
            elif pa[1] >= pa[0] and pa[1] >= pa[2]: nn[1] = np.copysign( 1, p[1] )
            elif pa[2] >= pa[0] and pa[2] >= pa[1]: nn[2] = np.copysign( 1, p[2] )
        pass
    pass
    return tt, nn



class Ray(object):
   def __init__(self, origin=[0,0,0], direction=[1,0,0] ):
       self.origin = np.asarray(origin, dtype=np.float32)
       dir_ = np.asarray(direction, dtype=np.float32)
       self.direction = dir_/np.sqrt(np.dot(dir_,dir_))   # normalize

   def position(self, tt):
       return self.origin + tt*self.direction

   def __repr__(self):
       o = self.origin
       d = self.direction
       return "Ray(o=[%5.2f,%5.2f,%5.2f], d=[%5.2f,%5.2f,%5.2f] )" % (o[0],o[1],o[2],d[0],d[1],d[2] )


   @classmethod
   def aringlight(cls, num=24, radius=500, center=[0,0,0], sign=-1., scale=1):
       rys = np.zeros( [num,2,3], dtype=np.float32 )

       a = np.linspace(0,2*np.pi,num )
       ca = np.cos(a)
       sa = np.sin(a)

       rys[:,0,0] = scale*radius*ca
       rys[:,0,1] = scale*radius*sa

       rys[:,0] += center

       rys[:,1,0] = sign*ca
       rys[:,1,1] = sign*sa

       return rys

   @classmethod
   def aboxlight(cls, num=24, side=500, center=[0,0,0], sign=-1., scale=3):
       a = np.linspace(-side,side,num )

       qys = np.zeros( [num,4,2,3], dtype=np.float32 )
       PX,MX,PY,MY = 0,1,2,3  
       for Q in [PX,MX,PY,MY]:
           if Q in [PX,MX]:
               qys[:,Q,0,0] = side*scale if Q == PX else -side*scale
               qys[:,Q,0,1] = a
               qys[:,Q,1,0] = sign if Q == PX else -sign 
               qys[:,Q,1,1] = 0
           elif Q in [PY,MY]:
               qys[:,Q,0,0] = a
               qys[:,Q,0,1] = side*scale if Q == PY else -side*scale
               qys[:,Q,1,0] = 0
               qys[:,Q,1,1] = sign if Q == PY else -sign 
           else:
               assert 0
           pass
       pass
       rys = qys.reshape(-1,2,3) 
       rys[:,0] += center
       return rys

   @classmethod
   def make_rays(cls, rys):
       rays = []
       for i in range(len(rys)):
           ray = cls(origin=rys[i,0], direction=rys[i,1])
           rays.append(ray)
       pass
       return rays
          
   @classmethod
   def ringlight(cls, num=24, radius=500, center=[0,0,0], sign=-1., scale=3):
       rys = cls.aringlight(num=num, radius=radius, center=center, sign=sign, scale=scale)
       return cls.make_rays(rys) 

   @classmethod
   def boxlight(cls, num=24, side=500, center=[0,0,0], sign=-1., scale=3):
       rys = cls.aboxlight(num=num, side=side, center=center, sign=sign, scale=scale)
       return cls.make_rays(rys) 

   @classmethod
   def origlight(cls, num=24):
      """
      :param num: of rays to create
      """
      return cls.ringlight(num=num, radius=0, sign=+1.)

   @classmethod
   def leaflight(cls, leaf, num=24, sign=-1., scale=3):
      """
      :param leaf: node
      :param num: number of rays, for boxlight get 4x rays
      :param sign: -1 for inwards, +1 for outwards
      :param scale: node radius or side to give ray origin position

      Rays based on leaf geometry 

      * inwards from outside the primitive: use scale ~ 3, and sign -1
      * outwards from inside the primitive: use scale ~ 0.1, and sign +1

      """
      if leaf.shape == SPHERE:
          return cls.ringlight(num=num, radius=leaf.param[3], center=leaf.param[:3], sign=sign, scale=scale)
      elif leaf.shape == BOX:
          return cls.boxlight(num=num, side=leaf.param[3], center=leaf.param[:3], sign=sign, scale=scale)
      else:
          pass
      return []





 
if __name__ == '__main__':

    plt.ion()
    plt.close()

    ii = test_ii_addseq()


if 0:
    cbox = Node(shape=BOX, param=[0,0,0,100] )
    csph = Node(shape=SPHERE, param=[0,0,0,100] )

    root2.annotate()


    #prim = cbox
    prim = csph


    num = 100

    ary = Ray.aringlight(num=num, radius=1000)

    rays = []
    rays += Ray.make_rays(ary)

    nray = len(rays)
    
    ii = np.zeros((nray, 4, 4 )) 
    for i, ray in enumerate(rays):
        tmin = 0 
        isect = intersect_primitive( prim, ray, tmin )
        ii[i] = isect[:]
        pass
    pass

    # repeat t to match shape of ray direction and add start position
    ipos = np.repeat( ii[:,0,3], 3).reshape(-1,3) * ii[:,3,:3] + ii[:,2,:3]
    ndir = ii[:,0,:3]

    print ipos
    print ndir

    sc = 10 
    plt.scatter( ipos[:,0]              , ipos[:,1] )
    plt.scatter( ipos[:,0]+ndir[:,0]*sc , ipos[:,1]+ndir[:,1]*sc )
    plt.scatter( ary[:,0,0], ary[:,0,1] )
    plt.show()



