#!/usr/bin/env python
"""
wavelength.py
===============

w0
    qudarap/tests/QCtxTest wavelength samples from GPU texture   

w1
    /tmp/G4OpticksAnaMgr/wavelength.npy from the horses mouth DsG4Scintillation 
    collected by G4OpticksAnaMgr at BeginOfRun
    Use ana/G4OpticksAnaMgr.sh to grab the outputs, see::

       jcv G4OpticksAnaMgr DsG4Scintillation
       extg4/tests/X4ScintillationIntegralTest.cc

w2
    np.interp results from the icdf obtained from GScintillatorLib, this
    closely matches w0 showing that the GPU texture lookups is working fine.

w3
    localSamples obtained from tests/X4ScintillationIntegralTest.cc
    using G4PhysicsOrderedFreeVector::GetEnergy interpolation.
    This clearly matches the horses mouth w1 in wavelength_cfplot.py 
    with chi2/ndf close to 1.
    

But w0 and w2 clearly exhibit some artifact of 20nm binning 
in the preparation of the icdf. The gross problem was fixed by avoiding 
the unintended standardization of raw material properties.


"""

import os, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.ana.key import keydir

class Wavelength(object):
    """
    Comparing localSamples with horsed

    """
    def __init__(self, kd):
        w = {}
        l = {}

        fold = "/tmp/QCtxTest"
        w[0] = np.load(os.path.join(fold, "wavelength.npy"))
        l[0] = "OK.QCtxTest"

        path1 = "/tmp/G4OpticksAnaMgr/WavelengthSamples.npy"
        w[1] = np.load(path1) if os.path.exists(path1) else None
        l[1] = "G4"
 
        aa = np.load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
        a = aa[0,:,0]
        b = np.linspace(0,1,len(a))
        u = np.random.rand(1000000)  
        w[2] = np.interp(u, b, a )  
        l[2] = "OK.GScint.interp"


        path2 = "/tmp/G4OpticksAnaMgr/localSamples.npy"
        w[3] = np.load(path2) if os.path.exists(path2) else None
        l[3] = "X4"
 
        #dom = np.arange(80, 800, 4)  
        #dom = np.arange(300, 600, 1)  
        dom = np.arange(350, 550, 1)  
        #dom = np.arange(385, 475, 1)  

        h = {}
        h[0],_ = np.histogram( w[0] , dom )
        h[1],_ = np.histogram( w[1] , dom )
        h[2],_ = np.histogram( w[2] , dom )
        h[3],_ = np.histogram( w[3] , dom )

        self.w = w  
        self.l = l
        self.h = h   
        self.dom = dom 
 


if __name__ == '__main__':
    kd = keydir(os.environ["OPTICKS_KEY"])
    wl = Wavelength(kd)

