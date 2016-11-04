#!/usr/bin/env python
"""
rayleigh.py 
=============================================

Without selection scatter distrib plots from 
arrays created by:

* optixrap/tests/ORayleighTest.cc
* cfg4/tests/OpRayleighTest.cc


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.nbase import vnorm, costheta_


OLDMOM,OLDPOL,NEWMOM,NEWPOL = 0,1,2,3
X,Y,Z=0,1,2



def dotmom_(a):
    oldmom = a[:,OLDMOM,:3]   
    newmom = a[:,NEWMOM,:3]   
    dotmom = costheta_(oldmom,newmom)
    return dotmom

def dotpol_(a):
    oldpol = a[:,OLDPOL,:3]   
    newpol = a[:,NEWPOL,:3]   
    dotpol = costheta_(oldpol,newpol)
    return dotpol


if __name__ == '__main__':

    aa = np.load(os.path.expandvars("$TMP/RayleighTest/ok.npy"))
    bb = np.load(os.path.expandvars("$TMP/RayleighTest/cfg4.npy"))

    bins = 100 
    nx = 4 
    ny = 2 

    qwns = [ 
         (1,aa[:,NEWMOM,X],bb[:,NEWMOM,X],"momx"), 
         (2,aa[:,NEWMOM,Y],bb[:,NEWMOM,Y],"momy"), 
         (3,aa[:,NEWMOM,Z],bb[:,NEWMOM,Z],"momz"), 
         (4,dotmom_(aa)   ,dotmom_(bb)   ,"dotmom"),

         (5,aa[:,NEWPOL,X],bb[:,NEWPOL,X],"polx"), 
         (6,aa[:,NEWPOL,Y],bb[:,NEWPOL,Y],"poly"), 
         (7,aa[:,NEWPOL,Z],bb[:,NEWPOL,Z],"polz"), 
         (8,dotpol_(aa)   ,dotpol_(bb)   ,"dotpol"),
           ]

    for i,a,b,label in qwns:
        plt.subplot(ny, nx, i)
        plt.hist(a, bins=bins, histtype="step", label=label)
        plt.hist(b, bins=bins, histtype="step", label=label)
    pass
    plt.show()


