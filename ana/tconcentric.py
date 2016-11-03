#!/usr/bin/env python
"""
tconcentric.py 
=============================================

Loads test events from Opticks and Geant4 and 
created by OKG4Test and 
compares their bounce histories.

"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.cf   import CF

STEP = 4

X,Y,Z=0,1,2


def butterfly_yz(plt, a, b, pt):
    plt.subplot(1, 2, 1)
    plt.scatter(a[:,pt,Y],a[:,pt,Z])

    plt.subplot(1, 2, 2)
    plt.scatter(b[:,pt,Y],b[:,pt,Z])


def butterfly(plt, scf):
    """
    Expecting 8cc6ccd 

              TO BT BT SC BT BT AB
              p0 p1 p2 p3 p4 p5 p6
                       
    p3: scatter occurs at point on X axis
    p4: first intersection point after the scatter  
    """
    a,b = scf.rpost()
    butterfly_yz(plt, a, b, pt=4)


if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="concentric")  

    log.info(ok.brief)

    cf = CF(ok)

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(0)

    spawn = ["8cc6ccd"]
    cf.init_spawn(spawn) 
    scf = cf.ss[0]

    #a,b = scf.rpost()
    a,b = scf.rpol()
    #a,b = scf.rw()


    

