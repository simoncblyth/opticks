#!/usr/bin/env python
"""
This turned out not to help much, easier to simply construct 
eye-look-up array directly as done in:

flightpath.py
    early attempts, at mpl viz of flight path with 3D arrows, 
    quiver plots turned out to be a better way  

mm0prim2.py 
    xz + xy circles path and testing flightpath cmds

"""
import numpy as np

class View(object):
    EYE = 0 
    LOOK = 1 
    UP = 2 
    def __init__(self, eye=[-1,1,0], look=[0,0,0], up=[0,0,1], dtype=np.float32 ):
        self.v = np.zeros( (4,4), dtype=dtype )
        self.eye = eye
        self.look = look
        self.up = up

    def _set_eye(self, a ):
        self.v[self.EYE,:3] = a[:3]
    def _get_eye(self):
        return self.v[self.EYE,:3] 
    eye = property(_get_eye, _set_eye)

    def _set_look(self, a ):
        self.v[self.LOOK,:3] = a[:3]
    def _get_look(self):
        return self.v[self.LOOK,:3] 
    look = property(_get_look, _set_look)

    def _set_up(self, a ):
        self.v[self.UP,:3] = a[:3]
    def _get_up(self):
        return self.v[self.UP,:3] 
    up = property(_get_up, _set_up)



if __name__ == '__main__':

     v = View()
     print repr(v.v)

     v.eye = [3,3,3]
     print repr(v.v)




