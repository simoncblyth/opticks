#!/usr/bin/env python
"""
wavelength.py
===============

0
    /tmp/G4OpticksAnaMgr/wavelength.npy from the horses mouth DsG4Scintillation 
    collected by G4OpticksAnaMgr at BeginOfRun
    Use ana/G4OpticksAnaMgr.sh to grab the outputs, see::

       jcv G4OpticksAnaMgr DsG4Scintillation
       extg4/tests/X4ScintillationIntegralTest.cc

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
    def get_key(self, label):
        key = None 
        for k,v in self.l.items(): 
            if v == label:
                key = k
            pass
        pass
        return key 

    def get_keys(self, a_label, b_label):
        a = self.get_key(a_label)
        b = self.get_key(b_label)
        return a, b 

    def __call__(self, label):
        return self.get_key(label) 

    def __init__(self, kd):
        p = {}
        l = {}

        l[0] = "DsG4Scintillator_G4OpticksAnaMgr"    ## horses mouth
        p[0] = "/tmp/G4OpticksAnaMgr/WavelengthSamples.npy"

        l[1] = "Opticks_QCtxTest_hd20"
        p[1] = os.path.join("/tmp/QCtxTest", "wavelength_20.npy")

        l[2] = "Opticks_QCtxTest_hd0"
        p[2] = os.path.join("/tmp/QCtxTest", "wavelength_0.npy")

        l[3] = "Opticks_QCtxTest_hd20_cudaFilterModePoint"
        p[3] = os.path.join("/tmp/QCtxTest", "wavelength_20_cudaFilterModePoint.npy")

        l[4] = "Opticks_QCtxTest_hd0_cudaFilterModePoint"
        p[4] = os.path.join("/tmp/QCtxTest", "wavelength_0_cudaFilterModePoint.npy")

        l[5] = "X4"
        p[5] = "/tmp/X4ScintillationTest/g4localSamples.npy"

        l[6] = "GScintillatorLib_np_interp"
        p[6] = os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy") 

        l[7] = "ck_photon"
        p[7] = os.path.join("/tmp/QCtxTest", "cerenkov_photon.npy")
      
        l[8] = "G4Cerenkov_modified_SKIP_CONTINUE"
        p[8] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_step_length_100000.000_SKIP_CONTINUE", "GenWavelength.npy")

        l[9] = "G4Cerenkov_modified_ASIS"
        p[9] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_step_length_100000.000_ASIS", "GenWavelength.npy")

 
        dom = np.arange(80, 400, 4)  
        #dom = np.arange(300, 600, 1)  
        #dom = np.arange(385, 475, 1)  


        #dom = np.arange(350, 550, 1)  


        a = {}
        w = {}
        h = {}
        for i in range(len(l)):
            if not os.path.exists(p[i]):
                a[i] = None
                w[i] = None
                h[i] = None
            else:
                a[i] = np.load(p[i])
                if l[i] == "ck_photon":
                    w[i] = a[i][:,0,1] 
                elif l[i].startswith("G4Cerenkov_modified"):
                    w[i] = a[i][:,0,1] 
                elif l[i] == "GScintillatorLib_np_interp":
                    aa = a[i] 
                    self.aa = aa
                    aa0 = aa[0,:,0]
                    bb0 = np.linspace(0,1,len(aa0))
                    u = np.random.rand(1000000)  
                    w[i] = np.interp(u, bb0, aa0 )  
                else:
                    w[i] = a[i]
                pass
                h[i] = np.histogram( w[i] , dom ) 
            pass
        pass
        self.p = p  
        self.w = w  
        self.l = l
        self.h = h   
        self.a = a   
        self.dom = dom 

    def interp(self, u):
        a = self.aa[0,:,0]
        b = np.linspace(0,1,len(a))
        return np.interp( u, b, a )
 


if __name__ == '__main__':
    kd = keydir(os.environ["OPTICKS_KEY"])
    wl = Wavelength(kd)

