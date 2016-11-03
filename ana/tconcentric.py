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
from opticks.ana.cf   import CF

if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="concentric")  

    log.info(ok.brief)

    spawn = ["8cc6ccd"]

    cf = CF(ok, spawn=spawn)

    scf = cf.ss[0]

    #a,b = scf.xyzt()
    a,b = scf.polw()




