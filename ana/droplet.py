#!/usr/bin/env python
"""
Droplet
~~~~~~~~~

Geometrical calculation of deviation, incident and refracted angles
at minimum deviation for k orders of rainbows.

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from opticks.ana.geometry import Boundary   
deg = np.pi/180.


class Droplet(object):
    def __init__(self, boundary):
        self.boundary = boundary 

    @classmethod
    def seqhis(cls, arg, src=None):
        pp = [arg] if type(arg) is int else arg 
        return map(lambda _:cls.seqhis_(_,src=src), pp) 

    @classmethod
    def seqhis_(cls, p, src=None):
        seq = "" if src is None else src + " "
        if p == 0:
            seq += "BR "
        elif p == 1:
            seq += "BT BT "
        elif p > 1:
            seq += "BT " + "BR " * (p-1) + "BT "  
        else:
            assert 0 
        pass
        seq += "SA"
        return seq 


    def deviation_angle(self, w, k=1): 
        d = self.deviation_angle_(w, k=k)
        return d['dv'] 

    def deviation_angle_(self, w, k=1): 
        """
        tot deviation, incident, refracted angles at the minimum deviation
        """
        if w is None:
            w = np.array([780., 380.])  # red, blue

        n = self.boundary.imat.refractive_index(w) 

        i = np.arccos( np.sqrt((n*n - 1.)/(k*(k+2.)) ))
        r = np.arcsin( np.sin(i)/n )
        dv = ( k*np.pi + 2*i - 2*r*(k+1) ) % (2*np.pi)

        return dict(n=n,i=i,r=r,dv=dv,k=k,w=w)

    def rainbow_table(self):
        redblue = np.array([780., 380.])
        lfmt = "%3s " + " %10s " * 7 
        print lfmt % ( "k", "th(red)", "th(blue)", "dth", "i(red)", "i(blue)", "r(red)", "r(blue)" )

        for k in range(1,21):
            d = self.deviation_angle_(redblue, k=k)
            dv = d['dv']
            i = d['i']
            r = d['r']
            fmt = "%3d " + " %10.2f " * 7 
            print fmt % (k, dv[0]/deg, dv[1]/deg, (dv[1]-dv[0])/deg, i[0]/deg, i[1]/deg, r[0]/deg, r[1]/deg ) 

    def bow_angle_rectangles(self, ks=range(1,11+1), yleg=2):
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        for k in ks:
            dvr = self.deviation_angle(None, k)/deg
            rect = Rectangle( (dvr[0], ymin), dvr[1]-dvr[0], ymax-ymin, alpha=0.1 ) 
            ax.add_patch(rect)
            ax.annotate( "%s" % k, xy=((dvr[0]+dvr[1])/2, yleg), color='red')
        pass




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    boundary = Boundary("Vacuum///MainH2OHale")
    droplet = Droplet(boundary)

    droplet.rainbow_table() 

    plt.ion()
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([0,4])
    ax.set_xlim([0,360])

    droplet.bow_angle_rectangles() 


     
