#!/usr/bin/env python
"""
wavelength.py
===============

w0
    qudarap/tests/QCtxTest wavelength samples from GPU texture   

w1
    /tmp/G4OpticksAnaMgr/wavelength.npy from the horses mouth DsG4Scintillation 
    collected by G4OpticksAnaMgr at BeginOfRun

w2
    np.interp results from the icdf obtained from GScintillatorLib, this
    closely matches w0 showing that the GPU texture lookups is working fine.

But w0 and w2 clearly exhibit some artifact of 20nm binning 
in the preparation of the icdf. 

TODO: look into GScintillatorLib is should be using all the information 
provided from material properties, not using coarse bins. 

"""
import os, numpy as np
from opticks.ana.key import keydir
from matplotlib import pyplot as plt 

if __name__ == '__main__':

    w = {}
    l = {}

    fold = "/tmp/QCtxTest"
    w[0] = np.load(os.path.join(fold, "wavelength.npy"))
    l[0] = "OK.QCtxTest"

    path1 = "/tmp/G4OpticksAnaMgr/WavelengthSamples.npy"
    w[1] = np.load(path1) if os.path.exists(path1) else None
    l[1] = "G4"

  
    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)
    aa = np.load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
    a = aa[0,:,0]
    b = np.linspace(0,1,len(a))
    u = np.random.rand(1000000)  
    w[2] = np.interp(u, b, a )  
    l[2] = "OK.GScint.interp"

    #bins = np.arange(80, 800, 4)  
    #bins = np.arange(300, 600, 1)  
    bins = np.arange(350, 550, 2)  

    h = {}
    h[0],_ = np.histogram( w[0] , bins )
    h[1],_ = np.histogram( w[1] , bins )
    h[2],_ = np.histogram( w[2] , bins )

    fig, ax = plt.subplots()

    fig.suptitle("Scintillation Wavelength Comparison")
 
    ax.plot( bins[:-1], h[0], drawstyle="steps-post", label=l[0] )  
    ax.plot( bins[:-1], h[1], drawstyle="steps-post", label=l[1] )  
    ax.plot( bins[:-1], h[2], drawstyle="steps-post", label=l[2] )  

    lines = False
    if lines:
        ylim = ax.get_ylim()
        for w in [320,340,360,380,400,420,440,460,480,500,520,540]:
            ax.plot( [w,w], ylim )    
        pass
    pass

    ax.legend()
    plt.show()



