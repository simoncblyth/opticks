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




def isolated_scatter():
    """
    Without selection scatter distrib plots 
    """ 
    OLDMOM,OLDPOL,NEWMOM,NEWPOL = 0,1,2,3
    aa = np.load(os.path.expandvars("$TMP/RayleighTest/ok.npy"))
    bb = np.load(os.path.expandvars("$TMP/RayleighTest/cfg4.npy"))

    bins = 100 
    nx = 6 
    ny = 2 

    qwns = [ 
         (1,aa[:,NEWMOM,X],"Amomx"), 
         (2,aa[:,NEWMOM,Y],"Amomy"), 
         (3,aa[:,NEWMOM,Z],"Amomz"), 
         (4,aa[:,NEWPOL,X],"Apolx"), 
         (5,aa[:,NEWPOL,Y],"Apoly"), 
         (6,aa[:,NEWPOL,Z],"Apolz"), 

         (7,bb[:,NEWMOM,X],"Bmomx"), 
         (8,bb[:,NEWMOM,Y],"Bmomy"), 
         (9,bb[:,NEWMOM,Z],"Bmomz"), 
         (10,bb[:,NEWPOL,X],"Bpolx"), 
         (11,bb[:,NEWPOL,Y],"Bpoly"), 
         (12,bb[:,NEWPOL,Z],"Bpolz"), 

           ]

    for i,q,label in qwns:
        plt.subplot(ny, nx, i)
        plt.hist(q, bins=bins, histtype="step", label=label)
    pass
    plt.show()



if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="concentric")  

    log.info(ok.brief)

    cf = CF(ok)

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(0)


    seq = "8cc6ccd"


    cf.init_spawn([seq]) 
    scf = cf.ss[0]

    fr0 = seq[::-1].find("6") - 1  # point before the SC
    fr1 = seq[::-1].find("8")      # SA 


    bins = 100 
    nx = 12 
    ny = fr1 - fr0 
    offset = 0

    log.info(" fr0 %d fr1 %d ny %d " % (fr0, fr1, ny))

    for fr in range(fr0,fr1):
        to = fr + 1

        adir, bdir = scf.rdir(fr, to)
        apol, bpol = scf.rpol_(fr)

        qwns = [ 
            (adir[:,X],"adirx"),
            (adir[:,Y],"adiry"),
            (adir[:,Z],"adirz"),
            (apol[:,X],"apolx"),
            (apol[:,Y],"apoly"),
            (apol[:,Z],"apolz"),

            (bdir[:,X],"bdirx"),
            (bdir[:,Y],"bdiry"),
            (bdir[:,Z],"bdirz"),
            (bpol[:,X],"bpolx"),
            (bpol[:,Y],"bpoly"),
            (bpol[:,Z],"bpolz"),
        ]

        for i,(q,label) in enumerate(qwns):
            plt.subplot(ny, nx, offset+i+1)
            plt.hist(q, bins=bins, histtype="step", label=label)
        pass
        offset += nx
    pass
    plt.show()








    

