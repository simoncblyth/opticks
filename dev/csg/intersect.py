#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from node import Node, SPHERE, BOX, EMPTY, UNION, INTERSECTION, DIFFERENCE, desc, root0, root1, root2, root3, root4

import logging
log = logging.getLogger(__name__)



class I(object):
   def __init__(self, t, n, name="", code=0):
       self.t = t
       self.n = n 
       self.name = name
       self.code = code
   def __repr__(self):
       return "I(%s [%s]) %s %d " % ( fmt_f(self.t), fmt_3f(self.n), self.name, self.code )


def intersect_primitive(node, ray, tmin):
    assert node.is_primitive
    #log.info("intersect_primitive") 
    if node.shape == BOX:
        tt, nn = intersect_box( node.param, ray, tmin )  
    elif node.shape == SPHERE:
        #log.info("intersect_primitive(SPHERE)") 
        tt, nn = intersect_sphere( node.param, ray, tmin)  
    elif node.shape == EMPTY:
        tt, nn = None, None
    else:
        log.fatal("shape unhandled shape:%s desc_shape:%s node:%s " % (node.shape, desc[node.shape], repr(node)))
        assert 0
    pass
    #print " intersect_node %s ray.direction %s tt %s nn %s " % ( desc[shape], repr(ray.direction), tt, repr(nn))
    return I(tt, nn, node.name, 0)



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
   def aringlight(cls, num=24, radius=500, inwards=True):
       rys = np.zeros( [num,2,3], dtype=np.float32 )

       a = np.linspace(0,2*np.pi,num )
       ca = np.cos(a)
       sa = np.sin(a)

       rys[:,0,0] = radius*ca
       rys[:,0,1] = radius*sa

       sign = -1. if inwards else +1.
       rys[:,1,0] = sign*ca
       rys[:,1,1] = sign*sa

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
   def ringlight(cls, num=24, radius=500, inwards=True):
       rys = cls.aringlight(num=num, radius=radius, inwards=inwards)
       return cls.make_rays(rys) 

   @classmethod
   def origlight(cls, num=24):
      """
      :param num: of rays to create
      """
      return cls.ringlight(num=num, radius=0, inwards=False)




 
if __name__ == '__main__':

    plt.ion()
    plt.close()

    cbox = Node(shape=BOX, param=[0,0,0,100] )
    csph = Node(shape=SPHERE, param=[0,0,0,100] )


    root2.tree_labelling()
    Node.dress(root2)


    #prim = cbox
    prim = csph


    num = 100

    ary = Ray.aringlight(num=num, radius=1000)

    rays = []
    rays += Ray.make_rays(ary)


    ipos = np.zeros((len(rays), 3), dtype=np.float32 ) 
    ndir = np.zeros((len(rays), 3), dtype=np.float32 ) 

    for i, ray in enumerate(rays):
        tmin = 0 
        tt, nn, nname, hmm = intersect_primitive( prim, ray, tmin )
        if not tt is None:
            ipos[i] = ray.position(tt)
            ndir[i] = nn
        pass
    pass
    print ipos
    print ndir

    sc = 10 
    plt.scatter( ipos[:,0]              , ipos[:,1] )
    plt.scatter( ipos[:,0]+ndir[:,0]*sc , ipos[:,1]+ndir[:,1]*sc )
    plt.scatter( ary[:,0,0], ary[:,0,1] )
    plt.show()



