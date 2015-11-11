#!/usr/bin/env python

import os, logging
import numpy as np
from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 
log = logging.getLogger(__name__)

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

class Evt(object):
    def __init__(self, tag="1", src="torch"):
        self.tag = str(tag)
        self.src = src
        self.ox = load_("ox"+src,tag) 
        self.rx = load_("rx"+src,tag) 
        self.ph = load_("ph"+src,tag)
        self.seqhis = self.ph[:,0,0]
        self.seqmat = self.ph[:,0,1]
        self.flags = self.ox.view(np.uint32)[:,3,3]
        self.fdom = np.load(idp_("OPropagatorF.npy"))

    def __repr__(self):
        return "Evt(%s,\"%s\")" % (self.tag, self.src)

    def history_table(self):
        seqhis = self.seqhis
        seqhis_table(count_unique(seqhis))

    def flags_table(self):
        flags = self.flags
        gflags_table(count_unique(flags))
        brsa = maskflags_int("BR|SA|TORCH")

    def seqhis_or(self, *args):
        """
        :param args: sequence strings excluding source, eg "BR SA" "BR AB"
        :return: selection boolean array of photon length

        photon level selection based on history sequence 
        """
        seqs = map(lambda _:"%s %s" % (self.src.upper(),_), args)
        s_seqhis = map(lambda _:self.seqhis == seqhis_int(_), seqs)
        return np.logical_or.reduce(s_seqhis)      

    def recpos(self, recs, irec):
        center = self.fdom[0,0,:3] 
        extent = self.fdom[0,0,3] 
        p = recs[:,irec,0,:3].astype(np.float32)*extent/32767.0 + center 
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

    def recpos(self, irec):
        return self.evt.recpos(self.rx, irec)



if __name__ == '__main__':
    e1 = Evt(1)




