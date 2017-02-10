#!/usr/bin/env python

import numpy as np
import logging
log = logging.getLogger(__name__)

SPHERE = 1
BOX = 2 
is_shape = lambda c:c in [SPHERE, BOX]

DIVIDER = 99  # between shapes and operations

UNION = 100
INTERSECTION = 101
DIFFERENCE = 102
is_operation = lambda c:c in [UNION,INTERSECTION,DIFFERENCE]


desc = { SPHERE:"SPHERE", BOX:"BOX", UNION:"UNION", INTERSECTION:"INTERSECTION", DIFFERENCE:"DIFFERENCE" }



def intersect_primitive(node, ray):
    assert node.is_primitive
    shape = node.left
    if shape == BOX:
        tt, nn = intersect_box( node.param, ray )  
    elif shape == SPHERE:
        tt, nn = intersect_sphere( node.param, ray )  
    else:
        log.fatal("shape unhandled shape:%s desc_shape:%s node:%s " % (shape, desc[shape], repr(node)))
        assert 0
    pass
    #print " intersect_node %s ray.direction %s tt %s nn %s " % ( desc[shape], repr(ray.direction), tt, repr(nn))
    return tt, nn


def intersect_sphere( param, ray ):

    center = param[:3]
    radius = param[3]

    O = ray.origin - center
    D = ray.direction

    b = np.dot(O, D)
    c = np.dot(O, O) - radius*radius

    disc = b*b - c
    sdisc = np.sqrt(disc) if disc > 0. else 0.

    root1 = -b - sdisc
    root2 = -b + sdisc

    has_intersect = sdisc > 0.

    if has_intersect:
        tt = root1 if root1 > ray.tmin else root2 
        nn = (O + tt*D)/radius 
    else:
        tt = None
        nn = None

    return tt, nn


def intersect_box( param, ray ):
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
 
    if has_intersect:
        if ray.tmin < t_near:
            tt = t_near 
        else:
            tt = t_far if ray.tmin < t_far else ray.tmin 

        p = ray.origin + tt*ray.direction - cen
        pa = np.abs(p)

        nn = np.array([0,0,0], dtype=np.float32)
        if   pa[0] >= pa[1] and pa[0] >= pa[2]: nn[0] = np.copysign( 1, p[0] )
        elif pa[1] >= pa[0] and pa[1] >= pa[2]: nn[1] = np.copysign( 1, p[1] )
        elif pa[2] >= pa[0] and pa[2] >= pa[1]: nn[2] = np.copysign( 1, p[2] )
    else:
        tt = None
        nn = None
    pass
    return tt, nn


class Node(object):
    def __init__(self, left, right=None, operation=None, param=None):

        self.left = left
        self.right = right
        self.operation = operation
        self.param = np.asarray(param) if not param is None else None 

        if not operation is None:
            left.parent = self 
            right.parent = self 
        pass

    is_primitive = property(lambda self:self.operation is None and self.right is None and not self.left is None)
    is_operation = property(lambda self:not self.operation is None)

    def __repr__(self):
        if self.is_primitive:
            return desc[self.left]
        else:
            return "%s(%s,%s)" % ( desc[self.operation], repr(self.left), repr(self.right) )

class Ray(object):
   def __init__(self, origin=[0,0,0], direction=[1,0,0], tmin=0 ):
       self.origin = np.asarray(origin, dtype=np.float32)
       dir_ = np.asarray(direction, dtype=np.float32)
       self.direction = dir_/np.sqrt(np.dot(dir_,dir_))   # normalize
       self.tmin = tmin

   def position(self, tt):
       return self.origin + tt*self.direction

   def __repr__(self):
       return "Ray(origin=%r, direction=%r, tmin=%s)" % (self.origin, self.direction, self.tmin )

   @classmethod
   def ringlight(cls, num=24, radius=500):
       angles = np.linspace(0,2*np.pi,num )

       ori = np.zeros( [num,3] )
       ori[:,0] = radius*np.cos(angles)
       ori[:,1] = radius*np.sin(angles)

       dir_ = np.zeros( [num,3] )
       dir_[:,0] = -np.cos(angles)
       dir_[:,1] = -np.sin(angles)

       rays = []
       for i in range(num):
           ray = cls(origin=ori[i], direction=dir_[i])
           rays.append(ray)
       pass
       return rays


 
if __name__ == '__main__':

    box = Node(BOX, param=[0,0,0,100] )
    sph = Node(SPHERE, param=[0,0,0,100] )

    axes = [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1],[1,1,1]]

    n = 100 
    angles = np.linspace(0,2*np.pi,n )
    dirs = np.zeros( [n,3] )
    dirs[:,0] = np.cos(angles)
    dirs[:,1] = np.sin(angles)
    
    for dir_ in dirs:
        ray = Ray(origin=[0,0,0], direction=dir_)
        #tt, nn = intersect_node( box, ray )
        tt, nn = intersect_node( sph, ray )





