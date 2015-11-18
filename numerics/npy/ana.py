#!/usr/bin/env python

import os, logging
import numpy as np
from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 
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


class Evt(object):
    def __init__(self, tag="1", src="torch", label=""):
        self.tag = str(tag)
        self.label = label
        self.src = src
        self.ox = load_("ox"+src,tag) 
        self.rx = load_("rx"+src,tag) 
        self.ph = load_("ph"+src,tag)
        self.seqhis = self.ph[:,0,0]
        self.seqmat = self.ph[:,0,1]
        self.flags = self.ox.view(np.uint32)[:,3,3]
        self.fdom = np.load(idp_("OPropagatorF.npy"))

    def msize(self):
        return float(self.ox.shape[0])/1e6

    def __repr__(self):
        return "Evt(%s,\"%s\",\"%s\")" % (self.tag, self.src, self.label)

    def history_table(self):
        seqhis = self.seqhis
        cu = count_unique(seqhis)
        seqhis_table(cu)
        tot = cu[:,1].astype(np.int32).sum()
        print "tot:", tot
        return cu

    def flags_table(self):
        flags = self.flags
        cu = count_unique(flags)
        gflags_table(cu)
        tot = cu[:,1].astype(np.int32).sum()
        print "tot:", tot
        brsa = maskflags_int("BR|SA|TORCH")
        return cu

    def seqhis_or(self, *args):
        """
        :param args: sequence strings excluding source, eg "BR SA" "BR AB"
        :return: selection boolean array of photon length

        photon level selection based on history sequence 
        """
        seqs = map(lambda _:"%s %s" % (self.src.upper(),_), args)
        s_seqhis = map(lambda _:self.seqhis == seqhis_int(_), seqs)
        return np.logical_or.reduce(s_seqhis)      

    def post_center_extent(self):
        p_center = self.fdom[0,0,:W] 
        p_extent = self.fdom[0,0,W] 

        t_center = self.fdom[1,0,X]
        t_extent = self.fdom[1,0,Y]
        
        center = np.zeros(4)
        center[:W] = p_center 
        center[W] = t_center

        extent = np.zeros(4)
        extent[:W] = p_extent*np.ones(3)
        extent[W] = t_extent

        return center, extent 

    def recpost(self, recs, irec):
        center, extent = self.post_center_extent()
        p = recs[:,irec,0].astype(np.float32)*extent/32767.0 + center 
        return p 



class Selection(object):
    def __init__(self, evt, *seqs):
        if len(seqs) > 0:
            psel = evt.seqhis_or(*seqs)
            rsel = np.repeat(psel, 10) # TODO: get these 10 from idom
            ox = evt.ox[psel]
            rx = evt.rx[rsel].reshape(-1,10,2,4)  
        else:
            psel = None
            rsel = None
            ox = evt.ox
            rx = evt.rx.reshape(-1,10,2,4)
        pass
        self.evt = evt 
        self.psel = psel
        self.rsel = rsel
        self.ox = ox
        self.rx = rx

    def recpost(self, irec):
        return self.evt.recpost(self.rx, irec)



if __name__ == '__main__':
    e1 = Evt(1)




