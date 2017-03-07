#!/usr/bin/env python

import numpy as np
import logging
from opticks.opticksnpy.NPart_h import EMPTY, ZERO, SPHERE, BOX

log = logging.getLogger(__name__)

X,Y,Z,W = 0,1,2,3


class RR(np.ndarray):
    """
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    ::
  
        0,0  ray.origin.x
        0,1  ray.origin.y
        0,2  ray.origin.z
        0,3  ray.tmin 

        1,0  ray.direction.x
        1,1  ray.direction.y
        1,2  ray.direction.z
        1,3  seq

    """
    O_ = 0, slice(0,3)
    TMIN_ = 0, 3
    D_ = 1, slice(0,3)
    SEQ_ = 1, 3

    def __new__(cls, a=None, history=[]):
        if a is None:
           a = np.zeros((2,4), dtype=np.float32 )
        pass
        obj = np.asarray(a).view(cls)
        obj.history = history
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.history = getattr(obj, 'history', None)

    def _get_tmin(self):
        return self[self.TMIN_[0],self.TMIN_[1]]
    def _set_tmin(self, t):
        self[self.TMIN_[0],self.TMIN_[1]] = t 
    tmin = property(_get_tmin, _set_tmin)

    def _get_o(self):
        return self[self.O_[0],self.O_[1]]
    def _set_o(self, o):
        self[self.O_[0],self.O_[1]] = o 
    o = property(_get_o, _set_o)
    origin = property(_get_o, _set_o)

    def _get_d(self):
        return self[self.D_[0],self.D_[1]]
    def _set_d(self, o):
        self[self.D_[0],self.D_[1]] = o 
    d = property(_get_d, _set_d)
    direction = property(_get_d, _set_d)

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


class RRS(np.ndarray):
    """
    Collection of rays
    """
    def __new__(cls, a=None, history=[]):
        if a is None:
           a = np.zeros((2,100,2,4), dtype=np.float32 )
        pass
        obj = np.asarray(a).view(cls)
        obj.history = history
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.history = getattr(obj, 'history', None)

    def rr(self, i):
        return self[i].view(RR)

    o = property(lambda self:self[:,RR.O_[0],RR.O_[1]])
    d = property(lambda self:self[:,RR.D_[0],RR.D_[1]])
    tmin = property(lambda self:self[:,RR.TMIN_[0],RR.TMIN_[1]])
    seq = property(lambda self:self.view(np.uint32)[:,RR.SEQ_[0],RR.SEQ_[1]])

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




class Ray(RR):
    def __init__(self, origin=[0,0,0], direction=[1,0,0], tmin=0):
       
        a = np.zeros( (2,4), dtype=np.float32 )
        RR.__init__(self, a)

        self.o = np.asarray(origin, dtype=np.float32)
        dir_ = np.asarray(direction, dtype=np.float32)
        self.d = dir_/np.sqrt(np.dot(dir_,dir_))   # normalize
        self.tmin = tmin 

    def position(self, tt):
        return self.o + tt*self.d

    def __repr__(self):
        o = self.o
        d = self.d
        return "Ray(o=[%5.2f,%5.2f,%5.2f], d=[%5.2f,%5.2f,%5.2f] )" % (o[0],o[1],o[2],d[0],d[1],d[2] )


    @classmethod
    def make_rys(cls, num):
        return RRS(np.zeros( [num,2,4], dtype=np.float32 )) 

    @classmethod
    def ringlight(cls, num=24, radius=500, center=[0,0,0], sign=-1., scale=1):
        rys = cls.make_rys(num)

        a = np.linspace(0,2*np.pi,num )
        ca = np.cos(a)
        sa = np.sin(a)

        rys[:,0,0] = scale*radius*ca
        rys[:,0,1] = scale*radius*sa

        rys[:,0,:3] += center

        rys[:,1,0] = sign*ca
        rys[:,1,1] = sign*sa

        return rys

    @classmethod
    def boxlight(cls, num=24, side=500, center=[0,0,0], sign=-1., scale=3):
        a = np.linspace(-side,side,num )

        qys = np.zeros( [num,4,2,4], dtype=np.float32 )
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
        rys = RRS(qys.reshape(-1,2,4)) 
        rys[:,0,:3] += center
        return rys

         
    @classmethod
    def origlight(cls, num=24):
         """
         :param num: of rays to create
         """
         return cls.ringlight(num=num, radius=0, sign=+1.)

    @classmethod
    def plane(cls, yside=100, offset=[0,0,0], num=10, direction=[1,0,0]):
        y = np.linspace(-yside, yside+1, num)
        rys = RRS(np.zeros( [num, 2, 4], dtype=np.float32 ))
        rys.o[:] = offset 
        rys.o[:,Y] += y 
        rys.d[:] = direction
        return rys 

    @classmethod
    def quad(cls, s=100, offset=[0,0,0], num=10, direction=[1,0,0]):
        r = np.linspace(-s,s+1,num)
        a = np.zeros( [num, 4, 2, 4], dtype=np.float32 )
        a[:,0,0,:3] = [side, 0, 0]
        return None
        #rays += [Ray(origin=[s,y,0], direction=[-1,0,0])  for y in r]
        #rays += [Ray(origin=[-s,y,0], direction=[1,0,0])  for y in r]
        #rays += [Ray(origin=[x,-s,0], direction=[0,1,0])  for x in r]
        #rays += [Ray(origin=[x, s,0], direction=[0,-1,0]) for x in r]


    @classmethod
    def seray(cls, s=300, num=100):
        v = np.linspace(-s,s+1,2)
        rys = RRS(np.zeros( [num, 2, 4], dtype=np.float32 ))
        rys.o[:,X] = v         
        rys.o[:,Y] = v - s
        rys.d[:,:3] = [-1,-1,0] 
        return rys
 

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
    ray = Ray()
    print ray  

    rl = Ray.ringlight()
    bl = Ray.boxlight()
    ol = Ray.origlight()


