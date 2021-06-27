#!/usr/bin/env python
import os, numpy as np
import matplotlib.pyplot as plt
from opticks.ana.key import keydir

def wavelengthSample(icdf, num_sample=1000000):
    u = np.random.rand(num_sample)
    w = np.interp( u, icdf.si, icdf.nm )
    return w  


class ScintillationIntegral(object):
    """
    aa,fc,sc 
        obtained from horses mouth DsG4Scintillation by G4OpticksAnaMgr
        at BeginOfRun and grabbed with ana/G4OpticksAnaMgr.sh 
    bb 
        integrated from fc by extg4/tests/X4ScintillationIntegralTest.cc

    """
    LABEL = "G4.ScintillationIntegral"
    FOLD = "/tmp/G4OpticksAnaMgr"

    def __init__(self):
        aa = np.load(os.path.join(self.FOLD,"ScintillationIntegral.npy"))  
        fc = np.load(os.path.join(self.FOLD,"FASTCOMPONENT.npy"))  
        sc = np.load(os.path.join(self.FOLD,"SLOWCOMPONENT.npy"))  
        assert np.all(fc == sc)   

        bb = np.load(os.path.join(self.FOLD,"X4ScintillationIntegralTest.npy")) 
        assert np.all(aa == bb)

        eV = aa[:,0]*1e6
        si = aa[:,1]/aa[:,1].max()

        nm = 1240./eV

        self.aa = aa
        self.bb = bb
        self.fc = fc
        self.sc = sc
        self.eV = eV
        self.nm = nm
        self.si = si 
        self.wl = wavelengthSample(self)


class GScintillator(object):
    LABEL = "OK.GScint"
    def __init__(self, kd):
        aa = np.load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
        fc = np.load(os.path.join(kd,"GScintillatorLib/LS/FASTCOMPONENT.npy"))
        sc = np.load(os.path.join(kd,"GScintillatorLib/LS/SLOWCOMPONENT.npy"))

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



def integral(sc):
    out = np.zeros((len(sc),2)) 
     
    currentIN = sc[0,1] 
    assert currentIN >= 0.0

    currentPM = sc[0,0]
    currentCII = 0.0; 

    out[0] = [currentPM, currentCII]

    prevPM  = currentPM
    prevCII = currentCII
    prevIN  = currentIN

    for ii in range(1,len(sc)):
        currentPM = sc[ii,0]
        currentIN = sc[ii,1]
        currentCII = 0.5 * (prevIN + currentIN)
        currentCII = prevCII + (currentPM - prevPM) * currentCII
        out[ii] = [currentPM, currentCII]
        prevPM  = currentPM
        prevCII = currentCII
    pass
    return out 




if __name__ == '__main__':
    
    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)

    gs = GScintillator(kd)
    si = ScintillationIntegral()

    #compare_icdf(gs, si)
    #compare_samples(gs, si)

    ff = integral(si.sc)


