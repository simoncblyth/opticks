#!/usr/bin/env python
"""
ana.py geometrical and plotting utils
========================================

TODO: reposition these into more appropriate locations

"""
import os, logging
import numpy as np

from opticks.ana.base import opticks_environment 
from opticks.ana.nbase import count_unique  
from opticks.ana.evt import Evt, costheta_

deg = np.pi/180.

log = logging.getLogger(__name__)

X,Y,Z,W = 0,1,2,3

def theta(xyz):
    """
    :param xyz: array of cartesian coordinates
    :return: array of Spherical Coordinare polar angle, theta in degrees

    First subtract off any needed translations to align
    the coordinate system with focal points.

    Spherical Coordinates (where theta is polar angle 0:pi, phi is azimuthal 0:2pi)

     x = r sin(th) cos(ph)  = r st cp   
     y = r sin(th) sin(ph)  = r st sp  
     z = r cos(th)          = r ct     


     sqrt(x*x + y*y) = r sin(th)
                  z  = r cos(th)

     atan( sqrt(x*x+y*y) / z ) = th 

    """
    r = np.linalg.norm(xyz, ord=2, axis=1)
    z = xyz[:,2]
    th = np.arccos(z/r)*180./np.pi
    return th


def scatter3d(fig,  xyz): 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

def histo(fig,  vals): 
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=91,range=[0,90])

def xyz3d(fig, path):
    xyz = np.load(path).reshape(-1,3)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])




class Rat(object):
    def __init__(self, n, d, label=""):
        n = len(n)
        d = len(d)
        r = float(n)/float(d)
        self.n = n
        self.d = d
        self.r = r
        self.label = label 
    def __repr__(self):
        return "Rat %s %s/%s %5.3f " % (self.label,self.n, self.d, self.r)


def recpos_plot(fig, evts, irec=0, nb=100, origin=[0,0,0] ):

    origin = np.asarray(origin)

    nr = len(evts)
    nc = 3
    clab = ["X","Y","Z"]

    for ir,evt in enumerate(evts):
        pos = evt.rpost_(irec)[:,:3] - origin 

        for ic, lab in enumerate(clab):
            ax = fig.add_subplot(nr,nc,1+ic+nc*ir)
            ax.hist(pos[:,ic],bins=nb)
            ax.set_xlabel(lab)


def angle_plot(fig, evts, irec=0, axis=[0,0,1], origin=[0,0,-200], nb=100):

    origin = np.asarray(origin)
    nc = len(evts)
    nr = 1

    for ic,evt in enumerate(evts):
        pos = evt.rpost_(irec)[:,:3] - origin

        axis_ = np.tile(axis, len(pos)).reshape(-1, len(axis))

        ct = costheta_(pos, axis_)
        th = np.arccos(ct)/deg

        ax = fig.add_subplot(nr,nc,1+ic)
        ax.hist(th, bins=nb) 
        ax.set_xlabel("angle to axis %s " % str(axis))




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    evt = Evt(tag="1", det="rainbow")



