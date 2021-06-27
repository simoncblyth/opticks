#!/usr/bin/env python
import os, numpy as np
import matplotlib.pyplot as plt

from opticks.ana.nload import np_load
from opticks.ana.key import keydir

def wavelengthSample(icdf, num_sample=1000000):
    u = np.random.rand(num_sample)
    w = np.interp( u, icdf.si, icdf.nm )
    return w  

class ScintillationIntegral(object):
    LABEL = "G4.ScintillationIntegral"
    PATH = "/tmp/G4OpticksAnaMgr/ScintillationIntegral.npy"
    def __init__(self):
        a = np.load(self.PATH)  
        eV = a[:,0]*1e6
        nm = 1240./eV
        si = a[:,1]/a[:,1].max()
 
        self.a = a 
        self.eV = eV
        self.nm = nm
        self.si = si 
        self.wl = wavelengthSample(self)

class GScintillator(object):
    LABEL = "OK.GScint"
    def __init__(self, kd):
        aa = np_load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
        fc = np_load(os.path.join(kd,"GScintillatorLib/LS/FASTCOMPONENT.npy"))
        sc = np_load(os.path.join(kd,"GScintillatorLib/LS/SLOWCOMPONENT.npy"))

        a = aa[0,:,0]
        b = np.linspace(0,1,len(a))

        self.nm = a 
        self.si = b 
        self.wl = wavelengthSample(self)



def compare_icdf(gs, si):
    fig = plt.figure()
    plt.title("Inverted Cumulative Distribution Function : for Scintillator Reemission " )
    ax = fig.add_subplot(1,1,1)

    ax.plot( gs.si, gs.nm, label=gs.LABEL) 
    ax.plot( si.si, si.nm, label=si.LABEL)

    ax.set_ylabel("Wavelength (nm)")
    ax.set_xlabel("Probability")
    ax.legend()
 
    fig.show()

def compare_samples(gs, si):

    bins = np.arange(300, 600, 2 )
    gsh = np.histogram( gs.wl, bins ) 
    sih = np.histogram( si.wl, bins ) 

    fig, ax = plt.subplots()
    ax.plot( bins[:-1], gsh[0], drawstyle="steps-post", label=gs.LABEL )
    ax.plot( bins[:-1], sih[0], drawstyle="steps-post", label=si.LABEL )
    ax.legend()

    fig.show()


if __name__ == '__main__':
    
    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)

    gs = GScintillator(kd)
    si = ScintillationIntegral()

    compare_icdf(gs, si)
    #compare_samples(gs, si)

